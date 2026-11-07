"""The data that moves through the pipeline and ends up in the JSON result.

All positions are [x0, y0, x1, y1] in pixels of that page's working image, with the origin at the
top-left. Each page records the width and height of that image, so a position can always be scaled.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]


class PageType(StrEnum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    SCANNED_BAD_TEXT = "scanned_bad_text"
    MIXED = "mixed"


class RegionType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"


class WordSource(StrEnum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class Word(BaseModel):
    text: str
    bbox: BBox
    source: WordSource
    confidence: float | None = None  # OCR confidence. None for text-layer words.
    recovered: bool = False  # True when the missed-text check (step 7) added this word.


class Cell(BaseModel):
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    bbox: BBox
    text: str = ""
    min_confidence: float | None = None
    words: list[Word] = Field(default_factory=list, exclude=True)


class Table(BaseModel):
    n_rows: int
    n_cols: int
    cells: list[Cell]
    rows: list[list[str]]  # Plain grid. A merged cell's text sits in its top-left position only.
    structure_source: str = "table_model"  # "table_model", or "word_alignment" if the model failed


class Region(BaseModel):
    type: RegionType
    bbox: BBox
    score: float
    text: str = ""
    min_confidence: float | None = None
    table: Table | None = None
    table_id: str | None = None  # Links to the joined table in DocumentResult.tables.
    image: str | None = None  # Chart or picture file, relative to the job's folder.
    labels: list[str] = Field(default_factory=list)  # Words read from a chart. Not chart data.
    extracted: bool = True  # False for charts: their values are saved as an image, never guessed.
    words: list[Word] = Field(default_factory=list, exclude=True)


class Page(BaseModel):
    number: int
    type: PageType
    width: int
    height: int
    deskew_degrees: float = 0.0
    rotated_degrees: int = 0
    missed_text_check_ran: bool = False
    regions: list[Region] = Field(default_factory=list)
    flagged_words: list[Word] = Field(default_factory=list)  # Words outside every region.
    notes: list[str] = Field(default_factory=list)


class JoinedTable(BaseModel):
    id: str
    pages: list[int]
    n_cols: int
    rows: list[list[str]]
    bboxes: list[BBox] = Field(default_factory=list)
    joined: bool = False
    cell_confidence_min: float | None = None
    row_sources: list[tuple[int, BBox]] = Field(default_factory=list, exclude=True)
    row_min_confidences: list[float | None] = Field(default_factory=list, exclude=True)


class ExtractedDocument(BaseModel):
    pages: list[Page]
    tables: list[JoinedTable] = Field(default_factory=list)


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class FieldValue(BaseModel):
    value: Any = None
    raw: str | None = None
    page: int | None = None
    position: BBox | None = None
    source: str | None = None  # "text_layer", "ocr", "mixed", "llm" or "person"
    ocr_confidence: float | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    # Checks that run once, outside the check module, such as "is it visible on the page" or "have we seen this
    # invoice before". They are copied into checks every time the checks run again.
    extra_checks: list[CheckResult] = Field(default_factory=list, exclude=True)
    confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    suggested_value: Any = None  # Claude's answer when it needs a person to confirm it.
    suggested_by: str | None = None
    alternatives: list[Any] = Field(default_factory=list, exclude=True)
    rule_notes: list[str] = Field(default_factory=list, exclude=True)  # Doubts the rules raised in step 13.
    evidence_penalty: float = Field(default=0.0, exclude=True)  # Lowers the score when evidence is indirect.
    llm_confirmed: bool = False  # Claude read the same value, which settles doubts about the reading only.
    recovery: str | None = None  # Set when a second pass over the line produced this value.


class LlmUsage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    models_used: list[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    fingerprint: str
    pipeline_version: str
    status: str  # "accepted" or "needs_review"
    document_type: str = "invoice"  # or "credit_note"
    page_count: int
    fields: dict[str, FieldValue]
    pages: list[Page]
    tables: list[JoinedTable]
    document_checks: list[CheckResult] = Field(default_factory=list)
    llm: LlmUsage = Field(default_factory=LlmUsage)
    seconds: float = 0.0
