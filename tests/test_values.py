from datetime import date
from decimal import Decimal

import pytest

from pdfpipe.fields.values import parse_date, parse_money


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("250.00", "250.00"),
        ("1,234.56", "1234.56"),
        ("1.234,56", "1234.56"),
        ("1 234.56", "1234.56"),
        ("$1,234.56", "1234.56"),
        ("1,234", "1234"),
        ("12,34", "12.34"),
        ("0.500", "0.500"),
        ("0,500", "0.500"),
        ("1.234.567", "1234567"),
        ("1,234,567", "1234567"),
        ("EUR 99.10", "99.10"),
        ("(50.00)", "-50.00"),
    ],
)
def test_reads_amounts(text: str, expected: str) -> None:
    assert parse_money(text) == Decimal(expected)


@pytest.mark.parametrize(
    "text",
    ["", "abc", "12.3.4x", "146.35 40.30", "123.45 678.90", "INV-1042", "1.234", "12.345", "$1.250"],
)
def test_refuses_things_that_are_not_one_amount(text: str) -> None:
    assert parse_money(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-05", date(2026, 3, 5)),
        ("05 Mar 2026", date(2026, 3, 5)),
        ("March 5, 2026", date(2026, 3, 5)),
        ("5th March 2026", date(2026, 3, 5)),
        ("25/03/2026", date(2026, 3, 25)),
        ("03/25/2026", date(2026, 3, 25)),
    ],
)
def test_reads_clear_dates(text: str, expected: date) -> None:
    assert parse_date(text) == (expected, False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # OCR often drops the space between a month and a number.
        ("February16, 2026", date(2026, 2, 16)),
        ("18 Jun2026", date(2026, 6, 18)),
        ("21Aug 2026", date(2026, 8, 21)),
        ("5 Mar. 2026", date(2026, 3, 5)),
        ("Sept 3, 2026", date(2026, 9, 3)),
        # Other languages.
        ("5. März 2026", date(2026, 3, 5)),
        ("5 mars 2026", date(2026, 3, 5)),
        ("5 août 2026", date(2026, 8, 5)),
        ("5 de marzo de 2026", date(2026, 3, 5)),
        ("5 settembre 2026", date(2026, 9, 5)),
        ("15 Dezember 2026", date(2026, 12, 15)),
        ("15.03.2026", date(2026, 3, 15)),
    ],
)
def test_reads_dates_with_missing_spaces_and_other_languages(text: str, expected: date) -> None:
    assert parse_date(text) == (expected, False)


def test_marks_a_date_that_could_be_either_way_as_ambiguous() -> None:
    _, ambiguous = parse_date("03/04/2026")
    assert ambiguous is True


def test_same_day_and_month_is_not_ambiguous() -> None:
    assert parse_date("04/04/2026") == (date(2026, 4, 4), False)


@pytest.mark.parametrize("text", ["16 Jul J 2026", "99/99/2026", "soon", "2026-13-40"])
def test_refuses_unreadable_dates(text: str) -> None:
    assert parse_date(text)[0] is None
