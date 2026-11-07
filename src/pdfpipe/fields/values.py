"""Turns printed dates and amounts into exact values, and refuses to guess when they are unclear."""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

# A space is allowed only as a thousands separator, as in 1 234.56. Without that limit, two separate
# amounts in one cell ("123.45 678.90") would read as one huge number.
_MONEY = re.compile(
    r"^[\(\-]?\s*(?:[$€£₹]|USD|EUR|GBP|INR)?\s*-?"
    r"(\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d[\d.,']*\d|\d)"
    r"\s*(?:[$€£₹]|USD|EUR|GBP|INR)?\)?$",
    re.IGNORECASE,
)
_MONTH_FORMATS = ["%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d-%b-%Y", "%d-%B-%Y", "%d %b, %Y", "%d %B, %Y"]
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$")
_ISO_DATE = re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$")


def parse_money(text: str) -> Decimal | None:
    """Reads an amount, handling both 1,234.56 and 1.234,56. Returns None if it isn't clearly one."""
    cleaned = text.strip()
    match = _MONEY.match(cleaned)
    if not match:
        return None
    digits = match.group(1).replace(" ", "").replace("'", "")
    last_dot, last_comma = digits.rfind("."), digits.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal_mark = "." if last_dot > last_comma else ","
    elif last_dot >= 0 or last_comma >= 0:
        mark = "." if last_dot >= 0 else ","
        decimals = len(digits) - digits.rfind(mark) - 1
        whole = digits[: digits.rfind(mark)]
        if digits.count(mark) == 1 and decimals == 3 and whole != "0":
            # 1,234 is 1234 in the usual English layout. But 1.234 is 1234 in many countries and 1.234 in
            # others, and nothing in the number says which. So it is left for a person.
            if mark == ".":
                return None
            decimal_mark = ""
        else:
            # 12,34 is a decimal comma. 0.500 is a decimal too, because a thousands group can't start with 0.
            decimal_mark = mark if digits.count(mark) == 1 else ""
    else:
        decimal_mark = ""
    if decimal_mark:
        whole, _, fraction = digits.rpartition(decimal_mark)
        whole = whole.replace(",", "").replace(".", "")
        normal = f"{whole}.{fraction}"
    else:
        normal = digits.replace(",", "").replace(".", "")
    try:
        value = Decimal(normal)
    except InvalidOperation:
        return None
    return -value if cleaned.startswith(("(", "-")) else value


def parse_date(text: str) -> tuple[date | None, bool]:
    """Returns (date, ambiguous). 03/04/2026 can be 3 April or March 4, so it's marked ambiguous."""
    cleaned = re.sub(r"\s+", " ", text.strip().rstrip(".,;")).replace(",", ", ").replace(" ,", ",")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cleaned)

    match = _ISO_DATE.match(cleaned)
    if match:
        try:
            return date(int(match[1]), int(match[2]), int(match[3])), False
        except ValueError:
            return None, False

    match = _NUMERIC_DATE.match(cleaned)
    if match:
        first, second, year = int(match[1]), int(match[2]), int(match[3])
        try:
            if first > 12:
                return date(year, second, first), False
            if second > 12:
                return date(year, first, second), False
            # Both readings are valid dates. Day-first is only a placeholder until a person checks.
            return date(year, second, first), first != second
        except ValueError:
            return None, False

    for variant in {cleaned, cleaned.replace(",", "")}:
        for fmt in _MONTH_FORMATS:
            try:
                return datetime.strptime(variant, fmt).date(), False
            except ValueError:
                continue
    return None, False
