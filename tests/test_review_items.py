"""Editing line items on the review page. The edited lines get the same checks the pipeline applies."""

from decimal import Decimal
from typing import Any

import pytest
from fastapi import HTTPException

from pdfpipe.service.app import MAX_ITEMS, ReviewRequest, apply_review, clean_items


def line(
    description: str = "Pens", quantity: Any = "2", unit_price: Any = "20.00", amount: Any = "40.00"
) -> dict[str, Any]:
    return {"description": description, "quantity": quantity, "unit_price": unit_price, "amount": amount}


def field(value: Any) -> dict[str, Any]:
    return {"value": value, "needs_review": True, "confidence": 0.5, "review_reasons": ["x"]}


def machine(**values: Any) -> dict[str, Any]:
    base = {
        "line_items": [line(), line("Paper", "3", "20.00", "60.00")],
        "subtotal": "100.00",
        "tax": "10.00",
        "grand_total": "110.00",
    }
    return {
        "document_type": values.pop("document_type", "invoice"),
        "fields": {k: field(v) for k, v in {**base, **values}.items()},
    }


def review(edited: Any, **machine_values: Any) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    request = ReviewRequest(edits=[{"name": "line_items", "value": edited}])  # type: ignore[list-item]
    return apply_review(machine(**machine_values), None, request)


class TestCleanItems:
    def test_a_valid_list_is_stored_with_exact_strings(self) -> None:
        cleaned = clean_items([line("  Pens  ", "2", "20", "40")])
        assert cleaned == [{"description": "Pens", "quantity": "2", "unit_price": "20", "amount": "40"}]

    def test_amounts_are_normalised(self) -> None:
        cleaned = clean_items([line(amount="1,234.50", unit_price="617.25", quantity="2")])
        assert cleaned is not None
        assert cleaned[0]["amount"] == "1234.50"

    def test_a_quantity_and_price_are_optional(self) -> None:
        cleaned = clean_items([line(quantity="", unit_price=None)])
        assert cleaned == [{"description": "Pens", "quantity": None, "unit_price": None, "amount": "40.00"}]

    @pytest.mark.parametrize("empty", [None, []])
    def test_clearing_the_list_is_allowed(self, empty: Any) -> None:
        assert clean_items(empty) is None

    @pytest.mark.parametrize(
        "bad",
        [
            [line(description="")],
            [line(description="x" * 301)],
            [line(amount="")],
            [line(amount="abc")],
            [line(amount="1.234")],  # could be 1234 or 1.234, so it isn't guessed
            [line(quantity="two")],
            [line(unit_price="-5")],
            [line(amount="-40.00")],
            ["not a line"],
            "not a list",
        ],
    )
    def test_a_bad_line_is_refused(self, bad: Any) -> None:
        with pytest.raises(HTTPException) as error:
            clean_items(bad)
        assert error.value.status_code == 422

    def test_the_message_names_the_line(self) -> None:
        with pytest.raises(HTTPException) as error:
            clean_items([line(), line(amount="abc")])
        assert "Line 2" in error.value.detail

    def test_negative_amounts_are_allowed_on_a_credit_note(self) -> None:
        cleaned = clean_items([line(amount="-40.00")], credit_note=True)
        assert cleaned is not None
        assert cleaned[0]["amount"] == "-40.00"

    def test_too_many_lines_are_refused(self) -> None:
        with pytest.raises(HTTPException):
            clean_items([line()] * (MAX_ITEMS + 1))


class TestEditedItemsAreChecked:
    def test_lines_that_add_up_are_accepted_and_recorded_as_a_person_edit(self) -> None:
        reviewed, changes = review([line("Pens", "2", "20.00", "40.00"), line("Ink", "3", "20.00", "60.00")])
        stored = reviewed["fields"]["line_items"]
        assert stored["source"] == "person"
        assert stored["needs_review"] is False
        assert [item["description"] for item in stored["value"]] == ["Pens", "Ink"]
        assert changes[0][0] == "line_items"
        assert changes[0][2] == stored["value"]  # the audit log keeps the new value

    def test_a_line_whose_quantity_times_price_is_wrong_is_refused(self) -> None:
        with pytest.raises(HTTPException) as error:
            review([line("Pens", "2", "20.00", "45.00"), line("Ink", "3", "20.00", "55.00")])
        assert "lines [1, 2]" in error.value.detail

    def test_lines_that_do_not_add_up_to_the_subtotal_are_refused(self) -> None:
        with pytest.raises(HTTPException) as error:
            review([line("Pens", "", "", "40.00"), line("Ink", "", "", "70.00")], subtotal="100.00")
        assert error.value.status_code == 422
        assert "subtotal" in error.value.detail

    def test_without_a_subtotal_the_lines_plus_tax_must_make_the_total(self) -> None:
        good = [line("Pens", "2", "20.00", "40.00"), line("Ink", "3", "20.00", "60.00")]
        reviewed, _ = review(good, subtotal=None)
        assert reviewed["fields"]["line_items"]["needs_review"] is False
        with pytest.raises(HTTPException) as error:
            review(good, subtotal=None, grand_total="150.00")
        assert "grand total" in error.value.detail

    def test_charges_are_part_of_the_sum(self) -> None:
        good = [line("Pens", "2", "20.00", "40.00"), line("Ink", "3", "20.00", "60.00")]
        reviewed, _ = review(good, subtotal=None, shipping="15.00", grand_total="125.00")
        assert reviewed["fields"]["line_items"]["needs_review"] is False

    def test_a_line_with_no_quantity_is_not_arithmetic_checked_but_still_summed(self) -> None:
        reviewed, _ = review([line("Service", "", "", "100.00")])
        assert reviewed["fields"]["line_items"]["needs_review"] is False
        with pytest.raises(HTTPException):
            review([line("Service", "", "", "99.00")])

    def test_a_credit_note_may_print_the_sign_either_way(self) -> None:
        negative = [line("Pens", "2", "20.00", "-40.00"), line("Ink", "3", "20.00", "-60.00")]
        reviewed, _ = review(
            negative, document_type="credit_note", subtotal="-100.00", tax="-10.00", grand_total="-110.00"
        )
        assert reviewed["fields"]["line_items"]["needs_review"] is False

    def test_a_rejected_edit_changes_nothing(self) -> None:
        original = machine()
        request = ReviewRequest(edits=[{"name": "line_items", "value": [line(amount="abc")]}])  # type: ignore[list-item]
        with pytest.raises(HTTPException):
            apply_review(original, None, request)
        assert original["fields"]["line_items"]["needs_review"] is True

    def test_the_printed_items_can_still_be_approved_when_the_document_itself_is_wrong(self) -> None:
        wrong = machine(subtotal="999.00")
        reviewed, _ = apply_review(wrong, None, ReviewRequest(approve=["line_items"]))
        assert reviewed["fields"]["line_items"]["needs_review"] is False
        assert reviewed["fields"]["line_items"]["approved_by_person"] is True

    def test_exact_arithmetic_is_used(self) -> None:
        # 0.1 + 0.2 is not 0.3 in floating point. It must be in this check.
        ok, _ = review(
            [line("a", "1", "0.10", "0.10"), line("b", "1", "0.20", "0.20")],
            subtotal="0.30",
            tax=None,
            grand_total="0.30",
        )
        assert Decimal(ok["fields"]["line_items"]["value"][0]["amount"]) + Decimal(
            ok["fields"]["line_items"]["value"][1]["amount"]
        ) == Decimal("0.30")
