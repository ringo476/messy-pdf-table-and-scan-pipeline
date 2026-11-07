"""Step 10: order regions the way a person reads the page, including across columns."""

from ..models import Region


def _cut(regions: list[Region]) -> list[Region]:
    """Recursive XY-cut: split at the widest empty band, across first, then down, and repeat."""
    if len(regions) <= 1:
        return regions

    def widest_gap(start: int, end: int) -> tuple[float, float | None]:
        spans = sorted((region.bbox[start], region.bbox[end]) for region in regions)
        reach = spans[0][1]
        best_gap, best_at = 0.0, None
        for low, high in spans[1:]:
            if low - reach > best_gap:
                best_gap, best_at = low - reach, (low + reach) / 2
            reach = max(reach, high)
        return best_gap, best_at

    gap_y, at_y = widest_gap(1, 3)
    gap_x, at_x = widest_gap(0, 2)
    if at_y is not None and (at_x is None or gap_y >= 0.5 * gap_x):
        above = [r for r in regions if (r.bbox[1] + r.bbox[3]) / 2 < at_y]
        below = [r for r in regions if (r.bbox[1] + r.bbox[3]) / 2 >= at_y]
        if above and below:
            return _cut(above) + _cut(below)
    if at_x is not None:
        left = [r for r in regions if (r.bbox[0] + r.bbox[2]) / 2 < at_x]
        right = [r for r in regions if (r.bbox[0] + r.bbox[2]) / 2 >= at_x]
        if left and right:
            return _cut(left) + _cut(right)
    return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))


def sort_regions(regions: list[Region]) -> list[Region]:
    return _cut(regions)
