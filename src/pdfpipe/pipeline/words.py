"""Step 6: get the words and their positions. The page type decides the source."""

import cv2
import numpy as np

from ..engines import Engines
from ..geometry import center, iou
from ..models import BBox, PageType, Region, Word, WordSource
from .classify import PageFacts

_CROP_PADDING = 12
_WHITE_BORDER = 16


def words_from_text_layer(facts: PageFacts, scale: float) -> list[Word]:
    """Converts text-layer words from PDF points to image pixels."""
    return [
        Word(
            text=str(item["text"]),
            bbox=(
                float(item["x0"]) * scale,
                float(item["top"]) * scale,
                float(item["x1"]) * scale,
                float(item["bottom"]) * scale,
            ),
            source=WordSource.TEXT_LAYER,
        )
        for item in facts.words
        if str(item["text"]).strip()
    ]


def ocr_box(image: np.ndarray, box: BBox, engines: Engines) -> list[Word]:
    """Runs OCR on one area of the page, and returns the words in page positions."""
    height, width = image.shape[:2]
    x0 = max(0, int(box[0]) - _CROP_PADDING)
    y0 = max(0, int(box[1]) - _CROP_PADDING)
    x1 = min(width, int(box[2]) + _CROP_PADDING)
    y1 = min(height, int(box[3]) + _CROP_PADDING)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    # A white margin helps the text detector with words that touch the edge of the crop.
    framed = cv2.copyMakeBorder(
        crop, _WHITE_BORDER, _WHITE_BORDER, _WHITE_BORDER, _WHITE_BORDER, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return engines.read_words(framed, offset=(float(x0 - _WHITE_BORDER), float(y0 - _WHITE_BORDER)))


def _sits_on_top_of(word: Word, other: Word) -> bool:
    """True when most of this word's width lies on the other word, on the same line."""
    width = max(1.0, word.bbox[2] - word.bbox[0])
    height = max(1.0, word.bbox[3] - word.bbox[1])
    shared_x = min(word.bbox[2], other.bbox[2]) - max(word.bbox[0], other.bbox[0])
    shared_y = min(word.bbox[3], other.bbox[3]) - max(word.bbox[1], other.bbox[1])
    return shared_x / width > 0.7 and shared_y / height > 0.4


def remove_duplicates(words: list[Word]) -> list[Word]:
    """The same letters can be read twice, because crops overlap. Keeps the surer reading.

    Longer words are kept first when confidence is close, so a stray letter can't replace a whole word.
    """
    ordered = sorted(
        words,
        key=lambda w: (w.confidence if w.confidence is not None else 1.0) + 0.02 * min(len(w.text), 10),
        reverse=True,
    )
    kept: list[Word] = []
    for word in ordered:
        if not any(iou(word.bbox, other.bbox) > 0.5 or _sits_on_top_of(word, other) for other in kept):
            kept.append(word)
    return kept


def ocr_regions(image: np.ndarray, regions: list[Region], engines: Engines) -> list[Word]:
    words: list[Word] = []
    for region in regions:
        words.extend(ocr_box(image, region.bbox, engines))
    return remove_duplicates(words)


def get_words(
    page_type: PageType,
    image: np.ndarray,
    regions: list[Region],
    facts: PageFacts,
    scale: float,
    engines: Engines,
) -> list[Word]:
    if page_type is PageType.DIGITAL:
        return words_from_text_layer(facts, scale)

    if page_type in (PageType.SCANNED, PageType.SCANNED_BAD_TEXT):
        return ocr_regions(image, regions, engines)

    # Mixed page: the text layer where it exists, and OCR on the embedded images.
    layer_words = words_from_text_layer(facts, scale)
    image_words: list[Word] = []
    for box in facts.image_boxes:
        pixel_box = (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)
        image_words.extend(ocr_box(image, pixel_box, engines))

    def already_in_layer(word: Word) -> bool:
        x, y = center(word.bbox)
        return any(w.bbox[0] <= x <= w.bbox[2] and w.bbox[1] <= y <= w.bbox[3] for w in layer_words)

    return layer_words + remove_duplicates([w for w in image_words if not already_in_layer(w)])
