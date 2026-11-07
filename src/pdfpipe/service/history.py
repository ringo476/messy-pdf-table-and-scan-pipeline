"""Uses a user's earlier invoices to check a new one.

Three things are checked against what was seen before from the same vendor:

- Duplicate: the same vendor and invoice number were already processed.
- Number shape: this vendor's numbers usually look like INV-#### and this one doesn't. This catches OCR mistakes
  that arithmetic can't, such as a letter read as a digit or a dropped digit.
- Currency: this vendor is usually invoiced in one currency and this one differs.

Only the same user's invoices are ever looked at. Nothing here changes a value. A check that fails adds a reason and
sends the field to a person, in the same way the checks in step 14 do.
"""

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

# A company's legal ending doesn't tell vendors apart, and it is often printed differently.
_LEGAL_ENDINGS = {
    "inc", "ltd", "llc", "llp", "gmbh", "ag", "sarl", "sas", "sl", "sa", "spa", "srl", "plc", "co", "corp",
    "corporation", "company", "limited", "pvt", "bv", "nv", "oy", "ab",
}  # fmt: skip
SAME_VENDOR_SIMILARITY = 0.92
MIN_HISTORY = 3  # A pattern needs at least this many earlier invoices.
MIN_AGREEMENT = 0.8  # And this share of them must agree before a different one counts as odd.


def vendor_key(name: str) -> str:
    """A vendor name reduced to what identifies it: no case, accents, punctuation or legal ending."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name.lower()) if not unicodedata.combining(c))
    words = re.sub(r"[^a-z0-9 ]+", " ", plain.replace("&", " and ")).split()
    while len(words) > 1 and words[-1] in _LEGAL_ENDINGS:
        words.pop()
    return " ".join(words)


def same_vendor(first: str, second: str) -> bool:
    return first == second or SequenceMatcher(None, first, second).ratio() >= SAME_VENDOR_SIMILARITY


def number_shape(number: str) -> str:
    """Each run of digits becomes #, so INV-1042 and INV-2210 have the same shape."""
    return re.sub(r"\d+", "#", number.strip())


def digit_count(number: str) -> int:
    return sum(character.isdigit() for character in number)


def _squash(number: str) -> str:
    return re.sub(r"[^a-z0-9]", "", number.lower())


def _flag(machine: dict[str, Any], field_name: str, check_name: str, detail: str) -> None:
    field = machine["fields"][field_name]
    field.setdefault("checks", []).append({"name": check_name, "passed": False, "detail": detail})
    field["needs_review"] = True
    field.setdefault("review_reasons", []).append(detail)
    field["confidence"] = min(float(field.get("confidence", 0.0)), 0.5)
    machine["status"] = "needs_review"


def _pass(machine: dict[str, Any], field_name: str, check_name: str, detail: str) -> None:
    machine["fields"][field_name].setdefault("checks", []).append(
        {"name": check_name, "passed": True, "detail": detail}
    )


def _value(machine: dict[str, Any], name: str) -> Any:
    return (machine.get("fields", {}).get(name) or {}).get("value")


def apply_history(machine: dict[str, Any], earlier: list[dict[str, Any]]) -> dict[str, Any]:
    """Checks a finished result against this vendor's earlier records. Returns the same dict, updated.

    earlier is every earlier record for this vendor (other jobs only), each with status, document_type,
    invoice_number, invoice_date, currency and grand_total.
    """
    number = _value(machine, "invoice_number")
    currency = _value(machine, "currency")
    document_type = machine.get("document_type", "invoice")
    same_type = [row for row in earlier if row["document_type"] == document_type]
    trusted = [row for row in same_type if row["status"] in ("accepted", "reviewed")]

    if number:
        for row in same_type:
            if row["invoice_number"] and _squash(row["invoice_number"]) == _squash(str(number)):
                when = row["invoice_date"].isoformat() if row["invoice_date"] else "an earlier date"
                total = _value(machine, "grand_total")
                if row["grand_total"] is not None and total is not None and str(row["grand_total"]) == str(total):
                    detail = f"this vendor's invoice '{number}' was already processed ({when}, same total)"
                else:
                    detail = (
                        f"this vendor's invoice '{number}' was already processed ({when}) "
                        f"with a different total ({row['grand_total']})"
                    )
                _flag(machine, "invoice_number", "not_a_duplicate", detail)
                break
        else:
            if same_type:
                _pass(
                    machine, "invoice_number", "not_a_duplicate", "no earlier invoice from this vendor has this number"
                )

        numbers = [row["invoice_number"] for row in trusted if row["invoice_number"]]
        if len(numbers) >= MIN_HISTORY:
            shapes = Counter(number_shape(n) for n in numbers)
            usual_shape, count = shapes.most_common(1)[0]
            if count / len(numbers) >= MIN_AGREEMENT:
                counts = [digit_count(n) for n in numbers]
                fewest, most = min(counts), max(counts)
                these = digit_count(str(number))
                if number_shape(str(number)) != usual_shape:
                    _flag(
                        machine,
                        "invoice_number",
                        "matches_vendor_pattern",
                        f"this vendor's invoice numbers usually look like {usual_shape}, but this one is '{number}'",
                    )
                elif these < fewest or these > most + 1:
                    # Numbers grow, so one more digit than ever before is normal. Fewer digits than any earlier
                    # number, or several more, usually means a digit was dropped or added by mistake.
                    usual = f"{fewest}" if fewest == most else f"{fewest} to {most}"
                    _flag(
                        machine,
                        "invoice_number",
                        "matches_vendor_pattern",
                        f"this vendor's invoice numbers usually have {usual} digits, but '{number}' has {these}",
                    )
                else:
                    _pass(
                        machine,
                        "invoice_number",
                        "matches_vendor_pattern",
                        f"looks like this vendor's usual {usual_shape}",
                    )

    currencies = [row["currency"] for row in trusted if row["currency"]]
    if currency and len(currencies) >= MIN_HISTORY:
        usual, count = Counter(currencies).most_common(1)[0]
        if count / len(currencies) >= MIN_AGREEMENT:
            if currency != usual:
                _flag(
                    machine,
                    "currency",
                    "matches_vendor_currency",
                    f"this vendor is usually invoiced in {usual}, but this invoice says {currency}",
                )
            else:
                _pass(machine, "currency", "matches_vendor_currency", f"this vendor is usually invoiced in {usual}")

    machine["history_checked"] = True
    machine["history_size"] = len(earlier)
    return machine
