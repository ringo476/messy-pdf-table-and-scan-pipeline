"""Steps 14 and 15: check each value with ordinary code, score it, and decide what needs a person."""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from ..config import Settings
from ..models import CheckResult, ExtractedDocument, FieldValue, PageType

REQUIRED_FIELDS = ("invoice_number", "invoice_date", "vendor_name", "grand_total", "line_items")
_MONEY_FIELDS = ("subtotal", "tax", "grand_total")
_DATE_FIELDS = ("invoice_date", "due_date")
INVOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_\-.]*$")
_LETTERS = re.compile(r"[A-Za-z]")
# A lowercase l beside a capital, or a letter that looks like a digit in the middle of digits.
_LOOKALIKE = re.compile(r"[A-Z]l|l[A-Z]|\d[OoIlSsB]\d|\b[0-9]+[OoIl]\b|\b[OoIl][0-9]{2,}\b")


def _add(field: FieldValue, name: str, passed: bool, detail: str = "") -> CheckResult:
    result = CheckResult(name=name, passed=passed, detail=detail)
    field.checks.append(result)
    return result


def _check_required(fields: dict[str, FieldValue]) -> None:
    for name in REQUIRED_FIELDS:
        field = fields[name]
        empty = field.value in (None, "", [])
        _add(field, "required_present", not empty, "this required field is empty" if empty else "")


def _check_formats(fields: dict[str, FieldValue]) -> None:
    number = fields["invoice_number"]
    if number.value is not None:
        _add(
            number,
            "valid_format",
            bool(INVOICE_ID.match(str(number.value))),
            "invoice numbers use letters, digits and - / _ .",
        )
    for name in _DATE_FIELDS:
        field = fields[name]
        if field.value is None:
            if field.raw:
                _add(field, "valid_format", False, f"could not read '{field.raw}' as a date")
            continue
        day = date.fromisoformat(str(field.value))
        _add(field, "valid_format", 2000 <= day.year <= 2100, "the date's year is outside 2000 to 2100")
    for name in _MONEY_FIELDS:
        field = fields[name]
        if field.value is not None:
            _add(
                field,
                "valid_format",
                isinstance(field.value, Decimal) and field.value >= 0,
                "amounts must not be negative",
            )
    vendor = fields["vendor_name"]
    if vendor.value is not None:
        _add(vendor, "valid_format", bool(_LETTERS.search(str(vendor.value))), "a vendor name must contain letters")
    invoice_date, due_date = fields["invoice_date"], fields["due_date"]
    if invoice_date.value and due_date.value:
        _add(
            due_date,
            "after_invoice_date",
            str(due_date.value) >= str(invoice_date.value),
            "the due date is before the invoice date",
        )


def _check_arithmetic(fields: dict[str, FieldValue], settings: Settings) -> list[CheckResult]:
    """Checks that each line, the subtotal and the grand total add up."""
    tolerance = Decimal(str(settings.money_tolerance))
    results: list[CheckResult] = []
    items_field = fields["line_items"]
    subtotal = fields["subtotal"].value
    tax = fields["tax"].value
    total = fields["grand_total"].value
    items: list[dict[str, Any]] = items_field.value or []

    # Line arithmetic: quantity x unit price = amount.
    if items:
        wrong, untestable = [], 0
        for number, item in enumerate(items, start=1):
            quantity, price, amount = item.get("quantity"), item.get("unit_price"), item.get("amount")
            if quantity is None or price is None or amount is None:
                untestable += 1
            elif abs(quantity * price - amount) > tolerance + Decimal("0.005"):
                wrong.append(number)
        if wrong:
            _add(
                items_field,
                "line_arithmetic",
                False,
                f"quantity x unit price doesn't equal the amount on lines {wrong[:10]}",
            )
        elif untestable:
            _add(items_field, "line_arithmetic", False, f"{untestable} lines are missing a quantity, price or amount")
        else:
            _add(items_field, "line_arithmetic", True)

    # Lines add up to the subtotal, or directly to the grand total if there is no subtotal.
    amounts: list[Decimal] = [item["amount"] for item in items if item.get("amount") is not None]
    if items and len(amounts) == len(items):
        items_sum = sum(amounts, Decimal(0))
        target_name = "subtotal" if subtotal is not None else "grand_total" if tax is None else None
        target = fields[target_name].value if target_name else None
        if target is not None:
            ok = abs(items_sum - target) <= tolerance
            detail = "" if ok else f"the line items add up to {items_sum}, but the {target_name} is {target}"
            _add(items_field, "lines_add_up", ok, detail)
            assert target_name is not None
            _add(fields[target_name], "lines_add_up", ok, detail)

    # Subtotal + tax = grand total.
    if subtotal is not None and total is not None:
        expected = subtotal + (tax if tax is not None else Decimal(0))
        ok = abs(expected - total) <= tolerance
        detail = "" if ok else f"subtotal + tax is {expected}, but the grand total is {total}"
        _add(fields["grand_total"], "total_adds_up", ok, detail)
        _add(fields["subtotal"], "total_adds_up", ok, detail)
        if tax is not None:
            _add(fields["tax"], "total_adds_up", ok, detail)
    elif total is not None and not items:
        _add(
            fields["grand_total"],
            "total_adds_up",
            False,
            "there are no line items or subtotal to check the total against",
        )
    return results


def _check_reading_quality(fields: dict[str, FieldValue], settings: Settings) -> None:
    """Applies only to OCR-read values, because text-layer words have no reading uncertainty."""
    for field in fields.values():
        if field.ocr_confidence is not None and not field.llm_confirmed:
            ok = field.ocr_confidence >= settings.min_ocr_confidence
            _add(field, "read_clearly", ok, f"OCR confidence is only {field.ocr_confidence:.2f}" if not ok else "")


def _check_lookalike_characters(fields: dict[str, FieldValue]) -> None:
    """Flags an OCR-read ID that mixes look-alike characters, such as 'Sl-123' for 'SI-123'.

    Arithmetic can't catch these, and one wrong character makes an invoice number useless. Text-layer
    values are exact, so they skip this check.
    """
    number = fields["invoice_number"]
    if number.value and number.source != "text_layer" and not number.llm_confirmed:
        suspicious = _LOOKALIKE.search(str(number.value))
        detail = f"OCR may have confused look-alike characters in '{suspicious.group(0)}'" if suspicious else ""
        _add(number, "no_lookalike_characters", suspicious is None, detail)


def _check_table_structure(fields: dict[str, FieldValue], document: ExtractedDocument) -> None:
    """Checks that every row fits the columns, and that amount columns hold numbers."""
    items_field = fields["line_items"]
    if not items_field.value:
        return
    table = next((t for t in document.tables if t.id == items_field.raw), None)
    if table is None:
        return
    uneven = [i for i, row in enumerate(table.rows) if len(row) != table.n_cols]
    unreadable = [i + 1 for i, item in enumerate(items_field.value) if item.get("amount") is None]
    _add(
        items_field,
        "table_structure",
        not uneven and not unreadable,
        f"rows that don't fit: {uneven[:5]}; unreadable amounts on lines: {unreadable[:5]}"
        if uneven or unreadable
        else "",
    )


def _check_ambiguity(fields: dict[str, FieldValue]) -> None:
    for field in fields.values():
        if field.suggested_value is not None:
            _add(
                field,
                "second_reader_agrees",
                False,
                f"Claude read this as '{field.suggested_value}', so a person needs to confirm it",
            )
        if field.alternatives:
            _add(
                field,
                "single_value",
                False,
                f"more than one value was found: {[str(v) for v in field.alternatives[:3]]}",
            )
        if field.value is not None or field.raw:
            for note in field.rule_notes:
                _add(field, "unambiguous", False, note)


def run_checks(fields: dict[str, FieldValue], document: ExtractedDocument, settings: Settings) -> None:
    """Runs every check. Re-running after a value changes starts from a clean slate."""
    for field in fields.values():
        field.checks = []
    _check_required(fields)
    _check_formats(fields)
    _check_arithmetic(fields, settings)
    _check_reading_quality(fields, settings)
    _check_lookalike_characters(fields)
    _check_table_structure(fields, document)
    _check_ambiguity(fields)


def score_and_decide(fields: dict[str, FieldValue], document: ExtractedDocument, settings: Settings) -> None:
    """Step 15. A field is accepted only if all its checks pass and its score meets the minimum.

    The score starts from how the words were read. A confirming sum raises it. A failed check caps
    it below any usable minimum, so a high score can never hide a failed check.
    """
    page_types = {page.number: page.type for page in document.pages}
    for name, field in fields.items():
        if field.value in (None, "", []) and not field.raw:
            required = name in REQUIRED_FIELDS
            field.confidence = 0.0
            failed_reasons = [check.detail or check.name for check in field.checks if not check.passed]
            field.needs_review = required or bool(failed_reasons)
            field.review_reasons = failed_reasons
            continue

        if field.source == "text_layer":
            base = 0.97
        elif field.source == "llm":
            base = 0.9
        elif field.ocr_confidence is not None:
            # OCR base: a strong reading gets near 0.9 before any confirmation.
            base = 0.9 * field.ocr_confidence
        else:
            base = 0.8
        if field.page is not None and page_types.get(field.page) is PageType.SCANNED_BAD_TEXT:
            base -= 0.02
        base -= field.evidence_penalty
        if field.llm_confirmed:
            base = max(base, 0.9)

        arithmetic_passed = any(
            check.passed and check.name in ("lines_add_up", "total_adds_up") for check in field.checks
        )
        if arithmetic_passed:
            base = min(0.995, base + 0.08)

        failed = [check for check in field.checks if not check.passed]
        field.confidence = round(min(base, 0.5) if failed else base, 3)
        reasons = [check.detail or check.name for check in failed]
        if not failed and field.confidence < settings.min_field_score:
            reasons.append(f"confidence {field.confidence:.2f} is below the minimum of {settings.min_field_score:.2f}")
        field.needs_review = bool(reasons)
        field.review_reasons = reasons
