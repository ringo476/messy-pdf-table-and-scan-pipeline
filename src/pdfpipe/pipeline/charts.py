"""Step 11: save each chart as an image and mark it as not extracted. Its values are never guessed."""

from pathlib import Path

import cv2
import numpy as np

from ..models import Region, RegionType


def save_charts(image: np.ndarray, regions: list[Region], page_number: int, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    index = 0
    for region in regions:
        if region.type is not RegionType.CHART:
            continue
        index += 1
        x0, y0, x1, y1 = (int(v) for v in region.bbox)
        crop = image[max(0, y0) : y1, max(0, x0) : x1]
        if crop.size == 0:
            continue
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        name = f"page{page_number}_chart{index}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if ok:
            (charts_dir / name).write_bytes(encoded.tobytes())
            region.image = f"charts/{name}"
