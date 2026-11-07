"""Turns printed dates and amounts into exact values, and refuses to guess when they are unclear."""

import re
import unicodedata
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


_MONTH_NAMES = {
    "jan": ("jan", "januar", "janvier", "enero", "gennaio", "janeiro", "januari"),
    "feb": ("feb", "februar", "fevrier", "febrero", "febbraio", "fevereiro", "februari"),
    "mar": ("mar", "marz", "maerz", "mrz", "mars", "marzo", "marco", "maart"),
    "apr": ("apr", "avril", "abril", "aprile"),
    "may": ("may", "mai", "mayo", "maggio", "maio", "mei"),
    "jun": ("jun", "juni", "juin", "junio", "giugno", "junho"),
    "jul": ("jul", "juli", "juillet", "julio", "luglio", "julho"),
    "aug": ("aug", "august", "aout", "agosto", "augustus"),
    "sep": ("sep", "sept", "september", "septembre", "septiembre", "setiembre", "settembre", "setembro"),
    "oct": ("oct", "okt", "oktober", "octobre", "octubre", "ottobre", "outubro"),
    "nov": ("nov", "november", "novembre", "noviembre", "novembro"),
    "dec": ("dec", "dez", "dezember", "decembre", "diciembre", "dicembre", "dezembro", "december"),
}
# English months aren't remapped, so "March" stays as it is. Only other languages need translating.
_MONTH_LOOKUP = {word: english for english, words in _MONTH_NAMES.items() for word in words}
_FILLER_WORDS = re.compile(r"\b(de|del|der|of|le|el|am)\b", re.IGNORECASE)


def _plain(token: str) -> str:
    """Lowercase and without accents, so 'Août' and 'aout' look the same."""
    decomposed = unicodedata.normalize("NFKD", token.lower())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _translate_months(text: str) -> str:
    def swap(match: re.Match[str]) -> str:
        token = match.group(0)
        plain = _plain(token)
        # Names English already understands (march, mar, may) are left alone.
        if plain in _ENGLISH_MONTH_WORDS:
            return token
        return _MONTH_LOOKUP.get(plain, token)

    return re.sub(r"[^\W\d_]+", swap, text)


_ENGLISH_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}  # fmt: skip


def parse_date(text: str) -> tuple[date | None, bool]:
    """Returns (date, ambiguous). 03/04/2026 can be 3 April or March 4, so it's marked ambiguous."""
    cleaned = re.sub(r"\s+", " ", text.strip().rstrip(".,;")).replace(",", ", ").replace(" ,", ",")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cleaned)
    # A full stop after a month ("5 Mar. 2026") and a missing space ("February16", "18 Jun2026") are common
    # in scans. Both are fixed before parsing, and the parse itself stays strict.
    cleaned = re.sub(r"(?<=[^\W\d_])\.", "", cleaned)
    cleaned = re.sub(r"(?<=\d)\.(?=\s*[^\W\d_])", "", cleaned)  # "5. März 2026"
    cleaned = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", cleaned)
    cleaned = _translate_months(cleaned)
    cleaned = _FILLER_WORDS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

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
