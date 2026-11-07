"""Step 2: classify each page. This decides where its words will come from in step 6."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..models import BBox, PageType

_WORDLIKE = re.compile(
    r"^[\(\[\"'#$€£₹]*[A-Za-z]{1,}[A-Za-z'\-]*[\)\]\"'.,:;!?%]*$|^[\(\[\"'#$€£₹\-+]*\d[\d.,:/\-]*[A-Za-z%)]{0,3}[.,:;]*$"
)
_VOWEL = re.compile(r"[aeiouyAEIOUY]")
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class PageFacts:
    """What the PDF itself says about a page, in PDF points."""

    width: float
    height: float
    words: list[dict[str, Any]] = field(default_factory=list)
    image_boxes: list[BBox] = field(default_factory=list)
    image_cover: float = 0.0


def read_page_facts(plumber_page: Any) -> PageFacts:
    width, height = float(plumber_page.width), float(plumber_page.height)
    words = plumber_page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False)
    boxes: list[BBox] = []
    for image in plumber_page.images:
        x0, y0 = max(0.0, float(image["x0"])), max(0.0, float(image["top"]))
        x1, y1 = min(width, float(image["x1"])), min(height, float(image["bottom"]))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return PageFacts(width, height, words, boxes, _cover(boxes, width, height))


def _cover(boxes: list[BBox], width: float, height: float) -> float:
    """The share of the page covered by images, counting overlaps once. Measured on a coarse grid."""
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    steps = 60
    covered = 0
    for i in range(steps):
        x = (i + 0.5) * width / steps
        for j in range(steps):
            y = (j + 0.5) * height / steps
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes):
                covered += 1
    return covered / (steps * steps)


def garbled_share(words: list[str]) -> float:
    """The share of tokens that don't look like real words or numbers."""
    if not words:
        return 0.0
    return sum(_looks_garbled(token) for token in words) / len(words)


def _looks_garbled(token: str) -> bool:
    if "(cid:" in token or "\N{REPLACEMENT CHARACTER}" in token or not _WORDLIKE.match(token):
        return True
    return len(token) >= 5 and token.isalpha() and not _VOWEL.search(token)


def text_agreement(layer_words: list[str], ocr_words: list[str]) -> float:
    """How much of what OCR sees on the page is also in the text layer, from 0 to 1."""
    ocr_tokens = _TOKEN.findall(" ".join(ocr_words).lower())
    if not ocr_tokens:
        return 1.0
    layer_tokens = set(_TOKEN.findall(" ".join(layer_words).lower()))
    return sum(token in layer_tokens for token in ocr_tokens) / len(ocr_tokens)


def classify_page(
    facts: PageFacts,
    settings: Settings,
    ocr_page_text: Callable[[], list[str]],
) -> tuple[PageType, str]:
    """Returns the page type and the reason for it.

    ocr_page_text is only called for a page that shows a full-page image and also has a text layer.
    That is the one case where reading the text alone can't tell a good layer from a bad one.
    """
    layer = [str(word["text"]) for word in facts.words]
    if len(layer) < settings.min_text_words:
        return PageType.SCANNED, "almost no text layer"
    if garbled_share(layer) > 1 - settings.garbled_text_limit:
        return PageType.SCANNED_BAD_TEXT, "text layer is garbled"
    if facts.image_cover >= settings.full_page_image_cover:
        agreement = text_agreement(layer, ocr_page_text())
        if agreement < settings.text_layer_min_agreement:
            return PageType.SCANNED_BAD_TEXT, f"text layer matches only {agreement:.0%} of the visible text"
        return PageType.DIGITAL, f"full-page image with a text layer that matches {agreement:.0%} of it"
    if facts.image_cover >= settings.mixed_image_cover:
        return PageType.MIXED, f"text layer plus images covering {facts.image_cover:.0%} of the page"
    return PageType.DIGITAL, "good text layer"
