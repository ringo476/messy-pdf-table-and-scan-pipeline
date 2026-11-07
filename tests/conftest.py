from pathlib import Path

import pytest

from pdfpipe.config import Settings
from pdfpipe.engines import Engines
from pdfpipe.models import ExtractedDocument, Page, PageType, Region, RegionType, Word, WordSource


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="session")
def engines() -> Engines:
    return Engines(Settings(_env_file=None))


def word(
    text: str, x0: float, y0: float, x1: float | None = None, y1: float | None = None, confidence: float | None = None
) -> Word:
    return Word(
        text=text,
        bbox=(x0, y0, x1 if x1 is not None else x0 + 10 * len(text), y1 if y1 is not None else y0 + 20),
        source=WordSource.OCR if confidence is not None else WordSource.TEXT_LAYER,
        confidence=confidence,
    )


def line_of(text: str, x: float, y: float, confidence: float | None = None) -> list[Word]:
    """Lays out words left to right on one line, with normal spacing."""
    words = []
    for token in text.split():
        words.append(word(token, x, y, confidence=confidence))
        x += 10 * len(token) + 8
    return words


def page_with_lines(lines: list[tuple[str, float, float]], number: int = 1, confidence: float | None = None) -> Page:
    words = [w for text, x, y in lines for w in line_of(text, x, y, confidence)]
    region = Region(type=RegionType.PARAGRAPH, bbox=(0, 0, 1200, 1700), score=0.9, words=words)
    return Page(number=number, type=PageType.DIGITAL, width=1200, height=1700, regions=[region])


def document_of(lines: list[tuple[str, float, float]], confidence: float | None = None) -> ExtractedDocument:
    return ExtractedDocument(pages=[page_with_lines(lines, confidence=confidence)])


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    return tmp_path
