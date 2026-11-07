"""Step 7: check for missed text, by comparing the words found with what the page image shows."""

import cv2
import numpy as np

from ..config import Settings
from ..engines import Engines
from ..geometry import overlap_of
from ..models import Region, RegionType, Word


def count_unread_blobs(image: np.ndarray, words: list[Word], regions: list[Region]) -> int:
    """Counts text-sized marks on the page that no word covers.

    Chart and picture regions are skipped, and long ruled lines are removed, so table borders and
    pictures don't count as missed text.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = ink.shape

    for size in ((max(25, width // 30), 1), (1, max(25, height // 30))):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))

    for word in words:
        x0, y0, x1, y1 = (int(v) for v in word.bbox)
        ink[max(0, y0 - 4) : y1 + 4, max(0, x0 - 4) : x1 + 4] = 0
    for region in regions:
        if region.type in (RegionType.CHART, RegionType.IMAGE):
            x0, y0, x1, y1 = (int(v) for v in region.bbox)
            ink[max(0, y0) : y1, max(0, x0) : x1] = 0

    # Merge the letters of one word into a single mark before counting.
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    scale = max(width, height) / 1700
    blobs = 0
    for index in range(1, count):
        blob_width, blob_height = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
        if 8 * scale <= blob_height <= 70 * scale and blob_width >= 14 * scale:
            blobs += 1
    return blobs


def empty_text_regions(words: list[Word], regions: list[Region]) -> int:
    empty = 0
    for region in regions:
        if region.type in (RegionType.CHART, RegionType.IMAGE):
            continue
        if not any(overlap_of(word.bbox, region.bbox) >= 0.5 for word in words):
            empty += 1
    return empty


def recover_missed_text(
    image: np.ndarray,
    words: list[Word],
    regions: list[Region],
    settings: Settings,
    engines: Engines,
) -> tuple[list[Word], bool, str | None]:
    """Returns all words, whether full-page OCR ran, and a note about what was recovered."""
    empty = empty_text_regions(words, regions)
    blobs = count_unread_blobs(image, words, regions)
    if empty == 0 and blobs < settings.missed_text_min_blobs:
        return words, False, None

    recovered: list[Word] = []
    for candidate in engines.read_words(image):
        covered = any(
            overlap_of(candidate.bbox, word.bbox) > 0.3 or overlap_of(word.bbox, candidate.bbox) > 0.5 for word in words
        )
        in_picture = any(
            region.type in (RegionType.CHART, RegionType.IMAGE) and overlap_of(candidate.bbox, region.bbox) > 0.5
            for region in regions
        )
        if not covered and not in_picture:
            recovered.append(candidate.model_copy(update={"recovered": True}))
    note = f"missed-text check ran ({empty} empty regions, {blobs} unread marks) and recovered {len(recovered)} words"
    return words + recovered, True, note
