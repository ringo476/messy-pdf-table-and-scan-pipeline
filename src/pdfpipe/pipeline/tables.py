"""Step 9: build each table. The table model finds the cells, and words go into cells by position."""

import logging
from collections import Counter

import numpy as np

from ..engines import DetectedCell, Engines
from ..geometry import box_distance, center, group_into_lines, min_confidence, union_box, words_to_text
from ..models import BBox, Cell, Region, RegionType, Table, Word

logger = logging.getLogger(__name__)

_CROP_PADDING = 8


def _grid_edges(count: int, spans: list[tuple[int, int, float, float, bool]]) -> list[float]:
    """Works out one set of column (or row) edges from all the cell boxes.

    Each span is (start index, span length, low, high, merged). Only unmerged cells vote, because
    merged-cell boxes from the model are the least reliable, in both directions.
    """
    lows: list[list[float]] = [[] for _ in range(count)]
    highs: list[list[float]] = [[] for _ in range(count)]
    for start, _, low, high, merged in spans:
        if not merged:
            lows[start].append(low)
            highs[start].append(high)
    # Merged cells only fill the gaps when a column or row has no unmerged cell at all.
    for start, length, low, high, _ in spans:
        if not lows[start]:
            lows[start].append(low)
        end = start + length - 1
        if not highs[end]:
            highs[end].append(high)

    edges: list[float | None] = [None] * (count + 1)
    edges[0] = float(np.median(lows[0])) if lows[0] else None
    edges[count] = float(np.median(highs[count - 1])) if highs[count - 1] else None
    for index in range(1, count):
        values = []
        if highs[index - 1]:
            values.append(float(np.median(highs[index - 1])))
        if lows[index]:
            values.append(float(np.median(lows[index])))
        edges[index] = sum(values) / len(values) if values else None

    known = [index for index, value in enumerate(edges) if value is not None]
    filled = np.interp(range(count + 1), known, [float(edges[index] or 0.0) for index in known])
    return [float(value) for value in np.maximum.accumulate(filled)]


def _column_edges(cells: list[DetectedCell], n_cols: int) -> list[float]:
    return _grid_edges(
        n_cols, [(c.col, c.colspan, c.bbox[0], c.bbox[2], c.rowspan > 1 or c.colspan > 1) for c in cells]
    )


def normalise_cells(cells: list[DetectedCell]) -> list[DetectedCell]:
    """Rebuilds every cell box from shared row and column edges, keeping the model's structure."""
    if not cells:
        return []
    n_rows = max(cell.row + cell.rowspan for cell in cells)
    n_cols = max(cell.col + cell.colspan for cell in cells)
    x_edges = _column_edges(cells, n_cols)
    y_edges = _grid_edges(
        n_rows, [(c.row, c.rowspan, c.bbox[1], c.bbox[3], c.rowspan > 1 or c.colspan > 1) for c in cells]
    )
    return [
        DetectedCell(
            bbox=(x_edges[c.col], y_edges[c.row], x_edges[c.col + c.colspan], y_edges[c.row + c.rowspan]),
            row=c.row,
            col=c.col,
            rowspan=c.rowspan,
            colspan=c.colspan,
        )
        for c in cells
    ]


def place_words_in_cells(words: list[Word], cells: list[DetectedCell]) -> dict[int, list[Word]]:
    """Puts every word in a cell. No word is dropped.

    All words on one visual line get the same row, decided by majority. This keeps a line together
    when a slight skew would otherwise push part of it into the row above or below. Within that row,
    each word goes to the column it overlaps most.
    """
    placed: dict[int, list[Word]] = {index: [] for index in range(len(cells))}
    if not cells or not words:
        return placed

    slots: dict[tuple[int, int], int] = {}
    for index, cell in enumerate(cells):
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                slots.setdefault((row, col), index)
    n_rows = max(cell.row + cell.rowspan for cell in cells)
    n_cols = max(cell.col + cell.colspan for cell in cells)

    def band(index: int, axis: int) -> tuple[float, float]:
        """The low and high edge of one row (axis 1) or column (axis 0)."""
        lows, highs = [], []
        for cell in cells:
            start, length = (cell.col, cell.colspan) if axis == 0 else (cell.row, cell.rowspan)
            if start == index:
                lows.append(cell.bbox[axis])
            if start + length - 1 == index:
                highs.append(cell.bbox[axis + 2])
        return (min(lows) if lows else 0.0, max(highs) if highs else 0.0)

    row_bands = [band(index, 1) for index in range(n_rows)]
    col_bands = [band(index, 0) for index in range(n_cols)]

    def nearest(value: float, bands: list[tuple[float, float]]) -> int:
        return min(
            range(len(bands)),
            key=lambda i: (
                0.0 if bands[i][0] <= value <= bands[i][1] else min(abs(value - bands[i][0]), abs(value - bands[i][1]))
            ),
        )

    for line in group_into_lines(words):
        votes = Counter(nearest(center(word.bbox)[1], row_bands) for word in line)
        row = votes.most_common(1)[0][0]
        for word in line:
            overlaps = [min(word.bbox[2], high) - max(word.bbox[0], low) for low, high in col_bands]
            col = int(np.argmax(overlaps))
            if overlaps[col] <= 0:
                col = nearest(center(word.bbox)[0], col_bands)
            target = slots.get((row, col))
            if target is None:
                target = min(range(len(cells)), key=lambda i: box_distance(word.bbox, cells[i].bbox))
            placed[target].append(word)
    return placed


def rows_disagree_with_lines(words: list[Word], cells: list[DetectedCell]) -> bool:
    """True when the model's rows can't explain the visual lines of text.

    This covers two errors. One is two lines landing in one row and competing for the same cells,
    which means the model missed a row. The other is a line outside every row, such as totals
    printed under the table. A wrapped description isn't an error, because only one cell repeats.
    """
    if not cells or not words:
        return False
    n_rows = max(cell.row + cell.rowspan for cell in cells)
    tops: dict[int, float] = {}
    bottoms: dict[int, float] = {}
    for cell in cells:
        tops[cell.row] = min(tops.get(cell.row, cell.bbox[1]), cell.bbox[1])
        end = cell.row + cell.rowspan - 1
        bottoms[end] = max(bottoms.get(end, cell.bbox[3]), cell.bbox[3])
    lines = group_into_lines(words)
    table_top, table_bottom = min(tops.values()), max(bottoms.values())
    occupied: dict[int, list[set[int]]] = {}
    for line in lines:
        y = sum(center(word.bbox)[1] for word in line) / len(line)
        if y < table_top or y > table_bottom:
            return True
        row = next((r for r in range(n_rows) if tops.get(r, table_top) <= y <= bottoms.get(r, table_top)), None)
        if row is None:
            continue
        columns = set()
        for word in line:
            x = center(word.bbox)[0]
            for cell in cells:
                if cell.row <= row < cell.row + cell.rowspan and cell.bbox[0] <= x <= cell.bbox[2]:
                    columns.add(cell.col)
        for earlier in occupied.get(row, []):
            if len(earlier & columns) >= 2:
                return True
        occupied.setdefault(row, []).append(columns)
    return False


def rows_from_lines(words: list[Word], cells: list[DetectedCell]) -> list[DetectedCell]:
    """Keeps the model's columns, but takes one row from each visual line of text."""
    n_cols = max(cell.col + cell.colspan for cell in cells)
    x_edges = _column_edges(cells, n_cols)
    # Words can stick out past the outer columns, so the outer edges are stretched to hold them.
    x_edges[0] = min(x_edges[0], min(word.bbox[0] for word in words))
    x_edges[-1] = max(x_edges[-1], max(word.bbox[2] for word in words))
    rebuilt = []
    for row, line in enumerate(group_into_lines(words)):
        top, bottom = min(w.bbox[1] for w in line), max(w.bbox[3] for w in line)
        for col in range(n_cols):
            rebuilt.append(
                DetectedCell(bbox=(x_edges[col], top, x_edges[col + 1], bottom), row=row, col=col, rowspan=1, colspan=1)
            )
    return rebuilt


def cells_from_alignment(words: list[Word]) -> list[DetectedCell]:
    """A backup used when the table model fails. Rows come from word height, columns from clear gaps."""
    if not words:
        return []
    heights = sorted(word.bbox[3] - word.bbox[1] for word in words)
    median_height = heights[len(heights) // 2]

    rows: list[tuple[float, float]] = []
    for word in sorted(words, key=lambda w: w.bbox[1]):
        if rows and word.bbox[1] < rows[-1][1] - 0.3 * median_height:
            rows[-1] = (rows[-1][0], max(rows[-1][1], word.bbox[3]))
        else:
            rows.append((word.bbox[1], word.bbox[3]))

    columns: list[tuple[float, float]] = []
    min_gap = 1.2 * median_height
    for word in sorted(words, key=lambda w: w.bbox[0]):
        if columns and word.bbox[0] < columns[-1][1] + min_gap:
            columns[-1] = (columns[-1][0], max(columns[-1][1], word.bbox[2]))
        else:
            columns.append((word.bbox[0], word.bbox[2]))

    return [
        DetectedCell(
            bbox=(col[0] - 1, row[0] - 1, col[1] + 1, row[1] + 1),
            row=row_index,
            col=col_index,
            rowspan=1,
            colspan=1,
        )
        for row_index, row in enumerate(rows)
        for col_index, col in enumerate(columns)
    ]


def _crop(image: np.ndarray, box: BBox) -> tuple[np.ndarray, tuple[float, float]]:
    height, width = image.shape[:2]
    x0 = max(0, int(box[0]) - _CROP_PADDING)
    y0 = max(0, int(box[1]) - _CROP_PADDING)
    x1 = min(width, int(box[2]) + _CROP_PADDING)
    y1 = min(height, int(box[3]) + _CROP_PADDING)
    return image[y0:y1, x0:x1], (float(x0), float(y0))


def build_table(image: np.ndarray, region: Region, engines: Engines) -> Table:
    crop, offset = _crop(image, region.bbox)
    source = "table_model"
    try:
        detected = engines.find_cells(crop, region.words, offset)
    except Exception:  # The table library can raise on unusual images, and a backup exists.
        logger.exception("table model failed, falling back to word alignment")
        detected = []
    # No cells, or a single row or column, means the model failed on this table.
    too_small = (
        max((c.row + c.rowspan for c in detected), default=0) < 2
        or max((c.col + c.colspan for c in detected), default=0) < 2
    )
    if too_small:
        detected = cells_from_alignment(region.words)
        source = "word_alignment"
    detected = normalise_cells(detected)
    if source == "table_model" and rows_disagree_with_lines(region.words, detected):
        detected = rows_from_lines(region.words, detected)
        source = "table_model_columns_with_line_rows"

    placed = place_words_in_cells(region.words, detected) if detected else {}
    n_rows = max((cell.row + cell.rowspan for cell in detected), default=0)
    n_cols = max((cell.col + cell.colspan for cell in detected), default=0)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    cells: list[Cell] = []
    for index, cell in enumerate(detected):
        words = placed.get(index, [])
        text = words_to_text(words).replace("\n", " ")
        if grid[cell.row][cell.col]:
            # Two detected cells share a slot. Keep both texts rather than lose one.
            text = f"{grid[cell.row][cell.col]} {text}".strip()
        grid[cell.row][cell.col] = text
        cells.append(
            Cell(
                row=cell.row,
                col=cell.col,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                bbox=union_box(w.bbox for w in words) if words else cell.bbox,
                text=text,
                min_confidence=min_confidence(words),
                words=words,
            )
        )
    return Table(n_rows=n_rows, n_cols=n_cols, cells=cells, rows=grid, structure_source=source)


def build_tables(image: np.ndarray, regions: list[Region], engines: Engines) -> list[str]:
    """Builds every table on a page. A 'table' with under two rows or columns becomes plain text."""
    notes: list[str] = []
    for region in regions:
        if region.type is not RegionType.TABLE:
            continue
        table = build_table(image, region, engines)
        if table.n_rows < 2 or table.n_cols < 2:
            region.type = RegionType.PARAGRAPH
            region.text = words_to_text(region.words)
            notes.append("a detected table had under two rows or columns, so it was kept as text")
            continue
        region.table = table
        region.text = "\n".join(" | ".join(row) for row in table.rows)
        region.min_confidence = min_confidence(region.words)
    return notes
