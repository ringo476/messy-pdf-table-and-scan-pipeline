"""Step 12: join a table that continues onto the next page, if its columns line up."""

from ..models import BBox, JoinedTable, Page, Region, RegionType

_BODY_TYPES = (RegionType.HEADING, RegionType.PARAGRAPH, RegionType.TABLE, RegionType.CHART, RegionType.IMAGE)
_COLUMN_TOLERANCE = 0.04


def _body(page: Page) -> list[Region]:
    return [region for region in page.regions if region.type in _BODY_TYPES]


def _column_starts(region: Region, page_width: int) -> list[float]:
    """The left edge of each column, as a share of the page width."""
    assert region.table is not None
    starts: list[list[float]] = [[] for _ in range(region.table.n_cols)]
    for cell in region.table.cells:
        if cell.colspan == 1 and cell.text:
            starts[cell.col].append(cell.bbox[0] / max(1, page_width))
    return [min(values) if values else -1.0 for values in starts]


def columns_line_up(first: Region, first_width: int, second: Region, second_width: int) -> bool:
    if first.table is None or second.table is None or first.table.n_cols != second.table.n_cols:
        return False
    for a, b in zip(_column_starts(first, first_width), _column_starts(second, second_width), strict=True):
        # Text can be right-aligned in a column, so compare only left-aligned first columns strictly.
        if a >= 0 and b >= 0 and abs(a - b) > _COLUMN_TOLERANCE * 3:
            return False
    first_box_x = first.bbox[0] / max(1, first_width)
    second_box_x = second.bbox[0] / max(1, second_width)
    return abs(first_box_x - second_box_x) <= _COLUMN_TOLERANCE


def _row_details(region: Region, page_number: int) -> tuple[list[tuple[int, BBox]], list[float | None]]:
    assert region.table is not None
    sources: list[tuple[int, BBox]] = []
    confidences: list[float | None] = []
    for row in range(region.table.n_rows):
        cells = [cell for cell in region.table.cells if cell.row == row]
        boxes = [cell.bbox for cell in cells] or [region.bbox]
        sources.append(
            (
                page_number,
                (
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ),
            )
        )
        values = [cell.min_confidence for cell in cells if cell.min_confidence is not None]
        confidences.append(min(values) if values else None)
    return sources, confidences


def _normalise(row: list[str]) -> list[str]:
    return [cell.strip().lower() for cell in row]


def join_tables(pages: list[Page]) -> list[JoinedTable]:
    """Builds the document's table list. Each table region links to its entry with table_id."""
    joined: list[JoinedTable] = []
    previous_last: tuple[Region, int, JoinedTable] | None = None

    for page in pages:
        body = _body(page)
        for position, region in enumerate(body):
            if region.type is not RegionType.TABLE or region.table is None:
                continue
            sources, confidences = _row_details(region, page.number)
            # The table model sometimes adds an empty row. Dropping it keeps later rows aligned.
            keep = [i for i, row in enumerate(region.table.rows) if any(text.strip() for text in row)]
            kept_rows = [region.table.rows[i] for i in keep]
            sources = [sources[i] for i in keep]
            confidences = [confidences[i] for i in keep]
            continues = (
                position == 0
                and previous_last is not None
                and previous_last[2].pages[-1] == page.number - 1
                and columns_line_up(previous_last[0], previous_last[1], region, page.width)
            )
            if continues and previous_last is not None:
                target = previous_last[2]
                rows = kept_rows
                # A header repeated on the new page isn't new data.
                skip = 1 if rows and target.rows and _normalise(rows[0]) == _normalise(target.rows[0]) else 0
                target.rows.extend(rows[skip:])
                target.row_sources.extend(sources[skip:])
                target.row_min_confidences.extend(confidences[skip:])
                target.pages.append(page.number)
                target.bboxes.append(region.bbox)
                target.joined = True
                region.table_id = target.id
                current = target
            else:
                current = JoinedTable(
                    id=f"table_{len(joined) + 1}",
                    pages=[page.number],
                    n_cols=region.table.n_cols,
                    rows=[list(row) for row in kept_rows],
                    bboxes=[region.bbox],
                    row_sources=sources,
                    row_min_confidences=confidences,
                )
                joined.append(current)
                region.table_id = current.id
            known = [value for value in current.row_min_confidences if value is not None]
            current.cell_confidence_min = min(known) if known else None
            if position == len(body) - 1:
                previous_last = (region, page.width, current)

        # A table can only continue if it was the last thing on the page before.
        if not body or body[-1].type is not RegionType.TABLE:
            previous_last = None
    return joined
