"""Tests for the page steps that need no models."""

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas

from conftest import word
from pdfpipe.config import Settings
from pdfpipe.engines import DetectedCell
from pdfpipe.models import Page as PageModel
from pdfpipe.models import PageType, Region, RegionType, Table
from pdfpipe.pipeline import assign, classify, imaging, intake, layout, reading, stitch, tables, words


def make_region(kind: RegionType, box: tuple[float, float, float, float], score: float = 0.9) -> Region:
    return Region(type=kind, bbox=box, score=score)


class TestIntake:
    def test_rejects_a_file_that_is_not_a_pdf(self, tmp_path: Path, settings: Settings) -> None:
        path = tmp_path / "fake.pdf"
        path.write_bytes(b"this is not a pdf at all")
        with pytest.raises(intake.RejectedFile, match="not a PDF"):
            intake.inspect_pdf(path, settings)

    def test_rejects_a_damaged_pdf(self, tmp_path: Path, settings: Settings) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.7\nnot really a pdf")
        with pytest.raises(intake.RejectedFile, match="damaged"):
            intake.inspect_pdf(path, settings)

    def test_rejects_an_empty_file(self, tmp_path: Path, settings: Settings) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        with pytest.raises(intake.RejectedFile, match="empty"):
            intake.inspect_pdf(path, settings)

    def test_rejects_a_password_protected_pdf(self, tmp_path: Path, settings: Settings) -> None:
        from reportlab.lib.pdfencrypt import StandardEncryption

        path = tmp_path / "locked.pdf"
        page = canvas.Canvas(str(path), encrypt=StandardEncryption("secret", canPrint=0))
        page.drawString(100, 100, "locked")
        page.save()
        with pytest.raises(intake.RejectedFile, match="password"):
            intake.inspect_pdf(path, settings)

    def test_rejects_too_many_pages(self, tmp_path: Path) -> None:
        path = tmp_path / "long.pdf"
        page = canvas.Canvas(str(path))
        for _ in range(4):
            page.drawString(100, 100, "x")
            page.showPage()
        page.save()
        with pytest.raises(intake.RejectedFile, match="limit"):
            intake.inspect_pdf(path, Settings(_env_file=None, max_pages=3))

    def test_same_content_gives_same_fingerprint(self, tmp_path: Path, settings: Settings) -> None:
        first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
        page = canvas.Canvas(str(first), invariant=1)
        page.drawString(100, 100, "same")
        page.save()
        second.write_bytes(first.read_bytes())
        assert intake.inspect_pdf(first, settings).fingerprint == intake.inspect_pdf(second, settings).fingerprint


class TestClassify:
    def facts(self, text: str, cover: float = 0.0) -> classify.PageFacts:
        return classify.PageFacts(600, 800, [{"text": t} for t in text.split()], [], cover)

    def never_called(self) -> list[str]:
        raise AssertionError("OCR should not run for this page")

    def test_good_text_layer_is_digital(self, settings: Settings) -> None:
        page_type, _ = classify.classify_page(
            self.facts("Invoice number 42 total due 100.00"), settings, self.never_called
        )
        assert page_type is PageType.DIGITAL

    def test_no_text_is_scanned(self, settings: Settings) -> None:
        page_type, _ = classify.classify_page(self.facts("", 1.0), settings, self.never_called)
        assert page_type is PageType.SCANNED

    def test_garbled_text_is_a_bad_scan(self, settings: Settings) -> None:
        text = "(cid:12)(cid:4) (cid:9)x �� qqq@@ ##%% zzzzz ^^^^ (cid:1)"
        page_type, _ = classify.classify_page(self.facts(text), settings, self.never_called)
        assert page_type is PageType.SCANNED_BAD_TEXT

    def test_full_page_image_with_wrong_layer_is_a_bad_scan(self, settings: Settings) -> None:
        page_type, _ = classify.classify_page(
            self.facts("the cat sat on a mat today", 1.0),
            settings,
            lambda: ["invoice", "total", "due", "paid", "amount"],
        )
        assert page_type is PageType.SCANNED_BAD_TEXT

    def test_full_page_image_with_matching_layer_uses_the_layer(self, settings: Settings) -> None:
        page_type, _ = classify.classify_page(
            self.facts("invoice total due paid amount", 1.0), settings, lambda: ["invoice", "total", "due", "paid"]
        )
        assert page_type is PageType.DIGITAL

    def test_text_plus_some_images_is_mixed(self, settings: Settings) -> None:
        page_type, _ = classify.classify_page(
            self.facts("invoice total due paid amount", 0.3), settings, self.never_called
        )
        assert page_type is PageType.MIXED


class TestCleanup:
    def text_image(self) -> np.ndarray:
        image = np.full((1000, 800, 3), 255, dtype=np.uint8)
        for row in range(12):
            cv2.putText(
                image,
                "invoice total amount paid today",
                (40, 80 + row * 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
            )
        return image

    @pytest.mark.parametrize("angle", [-4.0, -1.5, 2.0, 5.0])
    def test_finds_the_skew_angle(self, angle: float) -> None:
        image = self.text_image()
        matrix = cv2.getRotationMatrix2D((400, 500), angle, 1.0)
        turned = cv2.warpAffine(image, matrix, (800, 1000), borderValue=(255, 255, 255))
        assert imaging.estimate_skew(turned) == pytest.approx(-angle, abs=0.4)

    def test_blank_page_has_no_skew(self) -> None:
        assert imaging.estimate_skew(np.full((500, 400, 3), 255, dtype=np.uint8)) == 0.0

    def test_rendering_respects_the_pixel_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "large.pdf"
        page = canvas.Canvas(str(path), pagesize=(4000, 4000))
        page.drawString(100, 100, "large")
        page.save()
        document = pdfium.PdfDocument(str(path))
        image, _ = imaging.render_page(document[0], 300, Settings(_env_file=None, max_render_pixels=1_000_000))
        assert image.shape[0] * image.shape[1] <= 1_010_000
        document.close()


class TestLayoutCleanup:
    def test_removes_text_inside_a_table_and_duplicates(self) -> None:
        kept = layout.clean_regions(
            [
                make_region(RegionType.TABLE, (0, 0, 500, 500)),
                make_region(RegionType.PARAGRAPH, (50, 50, 200, 100)),
                make_region(RegionType.PARAGRAPH, (0, 600, 500, 700), 0.9),
                make_region(RegionType.PARAGRAPH, (2, 602, 500, 700), 0.5),
            ]
        )
        assert [region.type for region in kept] == [RegionType.TABLE, RegionType.PARAGRAPH]


class TestDuplicates:
    def test_removes_a_stray_letter_read_on_top_of_a_word(self) -> None:
        result = words.remove_duplicates(
            [word("Orbit", 35, 10, 163, 40, confidence=0.99), word("t", 136, 12, 159, 42, confidence=0.8)]
        )
        assert [w.text for w in result] == ["Orbit"]

    def test_keeps_neighbouring_words(self) -> None:
        result = words.remove_duplicates(
            [word("Grand", 0, 10, 50, 30, confidence=0.9), word("Total", 56, 10, 100, 30, confidence=0.9)]
        )
        assert len(result) == 2


class TestAssign:
    def test_every_word_goes_to_its_region_or_is_flagged(self) -> None:
        heading = make_region(RegionType.HEADING, (0, 0, 500, 50))
        table = make_region(RegionType.TABLE, (0, 100, 500, 300))
        chart = make_region(RegionType.CHART, (0, 400, 500, 600))
        flagged = assign.assign_words(
            [word("Title", 10, 10), word("Amount", 10, 150), word("Q1", 10, 450), word("Lost", 10, 700)],
            [heading, table, chart],
        )
        assert [w.text for w in heading.words] == ["Title"]
        assert [w.text for w in table.words] == ["Amount"]
        assert [w.text for w in chart.words] == ["Q1"]
        assert [w.text for w in flagged] == ["Lost"]

    def test_a_table_wins_over_a_paragraph_that_overlaps_it(self) -> None:
        paragraph = make_region(RegionType.PARAGRAPH, (0, 0, 500, 500))
        table = make_region(RegionType.TABLE, (0, 100, 500, 300))
        assign.assign_words([word("cell", 10, 150)], [paragraph, table])
        assert table.words and not paragraph.words

    def test_chart_words_become_labels(self) -> None:
        chart = make_region(RegionType.CHART, (0, 0, 500, 500))
        chart.words = [word("Q1", 10, 10), word("Q2", 50, 10)]
        assign.finish_region_text([chart])
        assert chart.labels == ["Q1", "Q2"]
        assert chart.text == ""


def grid_cells(rows: int, cols: int, row_height: float = 40, col_width: float = 100) -> list[DetectedCell]:
    return [
        DetectedCell(
            bbox=(c * col_width, r * row_height, (c + 1) * col_width, (r + 1) * row_height),
            row=r,
            col=c,
            rowspan=1,
            colspan=1,
        )
        for r in range(rows)
        for c in range(cols)
    ]


class TestTables:
    def test_a_wrong_merged_cell_box_is_rebuilt_from_the_grid(self) -> None:
        cells = [c for c in grid_cells(3, 2) if not (c.col == 0 and c.row < 2)]
        # The model got this merged cell's structure right, but its box far off to the right.
        cells.append(DetectedCell(bbox=(70, 0, 190, 80), row=0, col=0, rowspan=2, colspan=1))
        fixed = next(c for c in tables.normalise_cells(cells) if c.rowspan == 2)
        assert fixed.bbox[0] == pytest.approx(0, abs=1)
        assert fixed.bbox[2] == pytest.approx(100, abs=1)
        assert fixed.bbox[3] == pytest.approx(80, abs=1)

    def test_a_line_stays_in_one_row_despite_slight_skew(self) -> None:
        cells = tables.normalise_cells(grid_cells(2, 3))
        # The right-hand word sits slightly into the row above.
        line = [word("Toner", 5, 50, 50, 70), word("2", 105, 46, 115, 66), word("91.00", 205, 34, 250, 54)]
        placed = tables.place_words_in_cells(line, cells)
        rows = {cells[index].row for index, placed_words in placed.items() if placed_words}
        assert rows == {1}

    def test_no_word_is_lost(self) -> None:
        cells = tables.normalise_cells(grid_cells(2, 2))
        sample = [word("a", 5, 5, 15, 25), word("b", 150, 50, 160, 70), word("far", 900, 900, 930, 920)]
        placed = tables.place_words_in_cells(sample, cells)
        assert sum(len(v) for v in placed.values()) == 3

    def test_totals_under_the_table_are_noticed(self) -> None:
        cells = tables.normalise_cells(grid_cells(2, 2))
        sample = [word("a", 5, 5, 15, 25), word("Total", 5, 300, 50, 320)]
        assert tables.rows_disagree_with_lines(sample, cells) is True

    def test_two_lines_competing_for_one_row_are_noticed(self) -> None:
        cells = tables.normalise_cells(grid_cells(1, 2, row_height=80))
        sample = [
            word("a", 5, 5, 15, 25),
            word("1.00", 105, 5, 140, 25),
            word("b", 5, 40, 15, 60),
            word("2.00", 105, 40, 140, 60),
        ]
        assert tables.rows_disagree_with_lines(sample, cells) is True

    def test_a_wrapped_description_is_not_a_problem(self) -> None:
        cells = tables.normalise_cells(grid_cells(1, 2, row_height=80))
        sample = [word("long", 5, 5, 40, 25), word("9.00", 105, 5, 140, 25), word("text", 5, 40, 40, 60)]
        assert tables.rows_disagree_with_lines(sample, cells) is False

    def test_alignment_backup_finds_rows_and_columns(self) -> None:
        sample = [word("Item", 0, 0), word("Cost", 300, 0), word("Pen", 0, 40), word("2.00", 300, 40)]
        cells = tables.cells_from_alignment(sample)
        assert max(c.row for c in cells) == 1
        assert max(c.col for c in cells) == 1


class TestReadingOrder:
    def test_reads_down_one_column_then_the_next(self) -> None:
        title = make_region(RegionType.HEADING, (0, 0, 1000, 50))
        left_top = make_region(RegionType.PARAGRAPH, (0, 100, 450, 300))
        left_bottom = make_region(RegionType.PARAGRAPH, (0, 320, 450, 500))
        right_top = make_region(RegionType.PARAGRAPH, (550, 100, 1000, 300))
        right_bottom = make_region(RegionType.PARAGRAPH, (550, 320, 1000, 500))
        order = reading.sort_regions([right_bottom, left_bottom, title, right_top, left_top])
        assert order == [title, left_top, left_bottom, right_top, right_bottom]


def table_region(rows: list[list[str]], top: float = 0, left: float = 100) -> Region:
    cells = [
        tables.Cell(
            row=r, col=c, bbox=(left + c * 100, top + r * 30, left + (c + 1) * 100, top + (r + 1) * 30), text=text
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    table = Table(n_rows=len(rows), n_cols=len(rows[0]), cells=cells, rows=rows)
    return Region(
        type=RegionType.TABLE, bbox=(left, top, left + 100 * len(rows[0]), top + 30 * len(rows)), score=0.9, table=table
    )


def page_of(number: int, regions: list[Region]) -> PageModel:
    return PageModel(number=number, type=PageType.DIGITAL, width=1000, height=1400, regions=regions)


class TestJoin:
    header = ["Item", "Amount"]

    def test_joins_when_columns_line_up_and_drops_the_repeated_header(self) -> None:
        pages = [
            page_of(1, [table_region([self.header, ["a", "1"]])]),
            page_of(2, [table_region([self.header, ["b", "2"]])]),
        ]
        joined = stitch.join_tables(pages)
        assert len(joined) == 1
        assert joined[0].rows == [self.header, ["a", "1"], ["b", "2"]]
        assert joined[0].pages == [1, 2]
        assert joined[0].joined is True

    def test_does_not_join_when_column_counts_differ(self) -> None:
        pages = [
            page_of(1, [table_region([self.header, ["a", "1"]])]),
            page_of(2, [table_region([["x", "y", "z"], ["1", "2", "3"]])]),
        ]
        assert len(stitch.join_tables(pages)) == 2

    def test_does_not_join_when_text_sits_between_the_tables(self) -> None:
        note = make_region(RegionType.PARAGRAPH, (100, 900, 500, 950))
        pages = [
            page_of(1, [table_region([self.header, ["a", "1"]]), note]),
            page_of(2, [table_region([self.header, ["b", "2"]])]),
        ]
        assert len(stitch.join_tables(pages)) == 2

    def test_does_not_join_when_the_table_moves_sideways(self) -> None:
        pages = [
            page_of(1, [table_region([self.header, ["a", "1"]], left=100)]),
            page_of(2, [table_region([self.header, ["b", "2"]], left=500)]),
        ]
        assert len(stitch.join_tables(pages)) == 2

    def test_drops_empty_rows(self) -> None:
        pages = [page_of(1, [table_region([self.header, ["", ""], ["a", "1"]])])]
        assert stitch.join_tables(pages)[0].rows == [self.header, ["a", "1"]]
