"""Small box helpers used by the word placement, table and reading order steps."""

from collections.abc import Iterable

from .models import BBox, Word


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def overlap_of(a: BBox, b: BBox) -> float:
    """The share of box a that lies inside box b, from 0 to 1."""
    box_area = area(a)
    return intersection(a, b) / box_area if box_area > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    shared = intersection(a, b)
    union = area(a) + area(b) - shared
    return shared / union if union > 0 else 0.0


def center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def union_box(boxes: Iterable[BBox]) -> BBox:
    items = list(boxes)
    return (
        min(b[0] for b in items),
        min(b[1] for b in items),
        max(b[2] for b in items),
        max(b[3] for b in items),
    )


def box_distance(a: BBox, b: BBox) -> float:
    """Shortest distance between two boxes. Zero if they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def clip_box(box: BBox, width: int, height: int) -> BBox:
    return (
        max(0.0, min(box[0], width)),
        max(0.0, min(box[1], height)),
        max(0.0, min(box[2], width)),
        max(0.0, min(box[3], height)),
    )


def group_into_lines(words: list[Word]) -> list[list[Word]]:
    """Groups words that share a horizontal line, top to bottom, each line left to right."""
    lines: list[list[Word]] = []
    tops_bottoms: list[tuple[float, float]] = []
    for word in sorted(words, key=lambda w: (center(w.bbox)[1], w.bbox[0])):
        word_top, word_bottom = word.bbox[1], word.bbox[3]
        word_height = max(1.0, word_bottom - word_top)
        placed = False
        # Only recent lines can match, because words arrive sorted by height on the page.
        for index in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
            top, bottom = tops_bottoms[index]
            shared = min(word_bottom, bottom) - max(word_top, top)
            if shared >= 0.5 * min(word_height, max(1.0, bottom - top)):
                lines[index].append(word)
                tops_bottoms[index] = (min(top, word_top), max(bottom, word_bottom))
                placed = True
                break
        if not placed:
            lines.append([word])
            tops_bottoms.append((word_top, word_bottom))
    return [sorted(line, key=lambda w: w.bbox[0]) for line in lines]


def words_to_text(words: list[Word]) -> str:
    return "\n".join(" ".join(word.text for word in line) for line in group_into_lines(words))


def min_confidence(words: list[Word]) -> float | None:
    values = [word.confidence for word in words if word.confidence is not None]
    return min(values) if values else None
