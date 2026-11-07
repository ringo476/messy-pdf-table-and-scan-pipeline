"""Step 5: find the regions on the page image. This step locates regions but reads no words."""

import numpy as np

from ..engines import Engines
from ..geometry import area, clip_box, iou, overlap_of
from ..models import Region, RegionType

LABEL_TO_TYPE = {
    "doc_title": RegionType.HEADING,
    "paragraph_title": RegionType.HEADING,
    "table": RegionType.TABLE,
    "chart": RegionType.CHART,
    "image": RegionType.IMAGE,
    "seal": RegionType.IMAGE,
    "header_image": RegionType.IMAGE,
    "footer_image": RegionType.IMAGE,
    "header": RegionType.HEADER,
    "footer": RegionType.FOOTER,
    "number": RegionType.FOOTER,
}
# A word inside two regions belongs to the one with the highest rank.
RANK = {
    RegionType.TABLE: 4,
    RegionType.CHART: 3,
    RegionType.IMAGE: 1,
}


def clean_regions(regions: list[Region]) -> list[Region]:
    """Removes duplicate boxes, and text boxes that sit inside a table or chart."""
    ordered = sorted(regions, key=lambda r: (RANK.get(r.type, 2), r.score), reverse=True)
    kept: list[Region] = []
    for region in ordered:
        duplicate = any(iou(region.bbox, other.bbox) > 0.6 for other in kept)
        swallowed = any(
            overlap_of(region.bbox, other.bbox) > 0.7 and other.type in (RegionType.TABLE, RegionType.CHART)
            for other in kept
        )
        if not duplicate and not swallowed and area(region.bbox) > 16:
            kept.append(region)
    return kept


def find_regions(image: np.ndarray, engines: Engines) -> list[Region]:
    height, width = image.shape[:2]
    regions = []
    for detected in engines.find_regions(image):
        region_type = LABEL_TO_TYPE.get(detected.label, RegionType.PARAGRAPH)
        regions.append(
            Region(
                type=region_type,
                bbox=clip_box(detected.bbox, width, height),
                score=detected.score,
                extracted=region_type not in (RegionType.CHART, RegionType.IMAGE),
            )
        )
    return clean_regions(regions)
