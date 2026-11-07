"""Step 8: put each word in the region it sits inside."""

from ..geometry import area, overlap_of, words_to_text
from ..models import Region, RegionType, Word
from .layout import RANK


def assign_words(words: list[Word], regions: list[Region]) -> list[Word]:
    """Fills each region's word list, and returns the words outside every region.

    When a word is inside two regions, a table or chart wins over text. Between regions of the same
    rank, the smallest one wins, because it is the more specific box.
    """
    flagged: list[Word] = []
    for word in words:
        candidates = [region for region in regions if overlap_of(word.bbox, region.bbox) >= 0.5]
        if not candidates:
            flagged.append(word)
            continue
        best = max(candidates, key=lambda region: (RANK.get(region.type, 2), -area(region.bbox)))
        best.words.append(word)
    return flagged


def finish_region_text(regions: list[Region]) -> None:
    """Sets the text of each region, except tables, which step 9 handles."""
    for region in regions:
        if region.type in (RegionType.CHART, RegionType.IMAGE):
            region.labels = [word.text for word in region.words]
        elif region.type is not RegionType.TABLE:
            region.text = words_to_text(region.words)
