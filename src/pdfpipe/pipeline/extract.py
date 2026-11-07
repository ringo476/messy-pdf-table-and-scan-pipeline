"""Runs steps 2 to 12 on every page of a PDF, in the agreed order."""

import logging
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from ..config import Settings
from ..engines import Engines
from ..geometry import min_confidence
from ..models import ExtractedDocument, Page, PageType, RegionType
from . import assign, charts, classify, imaging, layout, missed, reading, stitch, tables, words

logger = logging.getLogger(__name__)


def process_page(
    plumber_page: object,
    pdfium_page: object,
    number: int,
    settings: Settings,
    engines: Engines,
    output_dir: Path | None,
) -> Page:
    facts = classify.read_page_facts(plumber_page)

    def ocr_for_classifier() -> list[str]:
        image, _ = imaging.render_page(pdfium_page, settings.digital_dpi, settings)
        return [word.text for word in engines.read_words(image)]

    # Step 2: classify the page.
    page_type, reason = classify.classify_page(facts, settings, ocr_for_classifier)

    # Step 3: convert the page to an image. Scanned pages get more pixels because OCR needs them.
    is_scan = page_type in (PageType.SCANNED, PageType.SCANNED_BAD_TEXT)
    image, scale = imaging.render_page(pdfium_page, settings.scan_dpi if is_scan else settings.digital_dpi, settings)
    notes = [f"classified as {page_type.value}: {reason}"]

    # Step 4: clean up scanned images only.
    deskew, rotated = 0.0, 0
    if is_scan:
        cleaned = imaging.clean_scan(image, engines)
        image, deskew, rotated = cleaned.image, cleaned.deskew_degrees, cleaned.rotated_degrees
        notes.extend(cleaned.notes)

    # Step 5: find the regions.
    regions = layout.find_regions(image, engines)

    # Step 6: get the words and their positions.
    page_words = words.get_words(page_type, image, regions, facts, scale, engines)

    # Step 7: check for missed text.
    page_words, check_ran, note = missed.recover_missed_text(image, page_words, regions, settings, engines)
    if note:
        notes.append(note)

    # Step 8: put each word in its region.
    flagged = assign.assign_words(page_words, regions)
    assign.finish_region_text(regions)
    for region in regions:
        region.min_confidence = min_confidence(region.words)

    # Step 9: build the tables.
    notes.extend(tables.build_tables(image, regions, engines))

    # Step 10: put the regions in reading order. The text inside each one was ordered in step 8.
    regions = reading.sort_regions(regions)

    # Step 11: save the charts.
    charts.save_charts(image, regions, number, output_dir)
    for region in regions:
        if region.type is RegionType.CHART:
            region.extracted = False

    height, width = image.shape[:2]
    return Page(
        number=number,
        type=page_type,
        width=width,
        height=height,
        deskew_degrees=deskew,
        rotated_degrees=rotated,
        missed_text_check_ran=check_ran,
        regions=regions,
        flagged_words=flagged,
        notes=notes,
    )


def extract_document(
    path: Path, settings: Settings, engines: Engines, output_dir: Path | None = None
) -> ExtractedDocument:
    pages: list[Page] = []
    pdfium_document = pdfium.PdfDocument(str(path))
    try:
        with pdfplumber.open(str(path)) as plumber_document:
            for index, plumber_page in enumerate(plumber_document.pages):
                pdfium_page = pdfium_document[index]
                try:
                    pages.append(process_page(plumber_page, pdfium_page, index + 1, settings, engines, output_dir))
                finally:
                    pdfium_page.close()
                    # pdfplumber caches every parsed object, which adds up on long documents.
                    plumber_page.flush_cache()
    finally:
        pdfium_document.close()

    # Step 12: join tables that continue across pages.
    joined = stitch.join_tables(pages)
    return ExtractedDocument(pages=pages, tables=joined)
