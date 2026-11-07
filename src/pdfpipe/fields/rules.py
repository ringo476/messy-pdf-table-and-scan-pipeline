"""Step 13: find the fields using labels, positions, table headers and formats.

Every value keeps its page and position. A field the rules can't find stays empty, and nothing is guessed.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..geometry import group_into_lines, overlap_of, union_box
from ..models import BBox, ExtractedDocument, FieldValue, JoinedTable, Page, RegionType, Word, WordSource
from .values import parse_date, parse_money


@dataclass
class Label:
    pattern: re.Pattern[str]
    rank: int = 0  # Lower rank wins when different labels both match.


def _labels(*patterns: tuple[str, int] | str) -> list[Label]:
    labels = []
    for item in patterns:
        pattern, rank = item if isinstance(item, tuple) else (item, 0)
        labels.append(Label(re.compile(rf"^{pattern}[:#.\s]*$", re.IGNORECASE), rank))
    return labels


# A tax rate after a label, as in "CGST 9%" or "VAT (20%)". It is part of the label, so the value is what follows.
_RATE = r"(?:\s*[@(]?\s*\d+(?:[.,]\d+)?\s*%\)?)?"

# English first, then German, French, Spanish and Italian. A label that isn't listed is never guessed: the
# field stays empty and goes to a person.
FIELD_LABELS = {
    "invoice_number": _labels(
        r"invoice\s*(?:no|number|num|#|id)\.?",
        r"inv\s*(?:no|#)\.?",
        r"bill\s*(?:no|number)\.?",
        r"rechnungs?[\s\-]*(?:nummer|nr|no)\.?",
        r"re[\s\-]*nr\.?",
        r"(?:num[eé]ro|n[°º]|no)\s*de\s*facture",
        r"facture\s*(?:n[°º]|no|num[eé]ro)\.?",
        r"n[úu]mero\s*de\s*factura",
        r"(?:n[°º]|no)\.?\s*(?:de\s*)?factura",
        r"factura\s*(?:n[°º]|no|n[úu]mero)\.?",
        r"(?:numero|n[°º]|no)\s*fattura",
        r"fattura\s*(?:n[°º]|no|numero)\.?",
    ),
    "invoice_date": _labels(
        r"invoice\s*date",
        r"bill\s*date",
        (r"date\s*of\s*issue", 0),
        (r"issue\s*date", 0),
        (r"date", 1),
        r"rechnungsdatum",
        (r"datum", 1),
        r"date\s*(?:de\s*)?(?:la\s*)?facture",
        r"date\s*d['’]?[ée]mission",
        r"fecha\s*(?:de\s*)?(?:factura|emisi[óo]n)",
        (r"fecha", 1),
        r"data\s*(?:della\s*)?fattura",
        (r"data", 1),
    ),
    "due_date": _labels(
        r"due\s*date",
        r"payment\s*due(?:\s*date)?",
        r"pay\s*by",
        r"f[äa]lligkeitsdatum",
        r"f[äa]llig\s*(?:am|bis)",
        r"zahlbar\s*bis",
        r"zahlungsziel",
        r"date\s*d['’]?[ée]ch[ée]ance",
        r"[ée]ch[ée]ance",
        r"fecha\s*(?:de\s*)?vencimiento",
        r"vencimiento",
        r"(?:data\s*)?scadenza",
    ),
    "vendor_name": _labels(
        r"vendor",
        r"seller",
        r"supplier",
        r"from",
        r"bill\s*from",
        r"lieferant",
        r"verk[äa]ufer",
        r"absender",
        r"fournisseur",
        r"vendeur",
        r"[ée]metteur",
        r"proveedor",
        r"vendedor",
        r"emisor",
        r"fornitore",
        r"venditore",
    ),
    "bill_to": _labels(
        r"bill\s*to",
        r"billed\s*to",
        r"customer",
        r"sold\s*to",
        r"rechnung\s*an",
        r"rechnungsempf[äa]nger",
        r"kunde",
        r"factur[ée]\s*[àa]",
        r"client",
        r"facturar\s*a",
        r"facturado\s*a",
        r"cliente",
        r"fatturare\s*a",
        r"intestata\s*a",
    ),
    "currency": _labels(r"currency", r"w[äa]hrung", r"devise", r"moneda", r"valuta"),
    "subtotal": _labels(
        r"sub[\s\-]*total",
        r"net\s*total",
        r"total\s*before\s*tax",
        r"net\s*amount",
        r"zwischensumme",
        r"nettobetrag",
        r"sous[\s\-]*total",
        r"total\s*h\.?t\.?",
        r"montant\s*h\.?t\.?",
        r"base\s*imponible",
        r"importe\s*neto",
        r"subtotale",
        r"(?:totale\s*)?imponibile",
    ),
    "tax": _labels(
        r"(?:sales\s*)?tax(?:\s*amount)?" + _RATE,
        r"vat(?:\s*amount)?" + _RATE,
        r"gst(?:\s*amount)?" + _RATE,
        r"total\s*tax",
        r"(?:mwst|ust|mehrwertsteuer|umsatzsteuer|steuer)\.?" + _RATE,
        r"(?:tva|taxe)" + _RATE,
        r"(?:iva|impuesto|imposta)" + _RATE,
    ),
    "grand_total": _labels(
        r"grand\s*total",
        r"total\s*(?:amount\s*)?due",
        r"amount\s*due",
        r"balance\s*due",
        r"invoice\s*total",
        (r"total(?:\s*amount)?", 1),
        r"gesamtbetrag",
        r"rechnungsbetrag",
        r"gesamtsumme",
        r"endbetrag",
        r"zu\s*zahlen",
        r"total\s*ttc",
        r"montant\s*ttc",
        r"net\s*[àa]\s*payer",
        r"total\s*[àa]\s*payer",
        r"montant\s*total",
        r"total\s*a\s*pagar",
        r"importe\s*total",
        r"total\s*factura",
        r"totale\s*fattura",
        r"totale\s*da\s*pagare",
        (r"totale", 1),
    ),
    # Charges and reductions between the subtotal and the total. Without them, a total with a shipping fee or
    # a discount could never add up, and a correct invoice would always go to a person.
    "discount": _labels(
        r"(?:total\s*)?discount" + _RATE,
        r"rebate",
        r"rabatt" + _RATE,
        r"nachlass",
        r"remise" + _RATE,
        r"r[ée]duction",
        r"descuento" + _RATE,
        r"sconto" + _RATE,
    ),
    "shipping": _labels(
        r"(?:shipping|freight|postage|handling)(?:\s*(?:and|&|/)\s*handling)?(?:\s*(?:charges?|costs?|fees?))?",
        r"delivery\s*(?:charges?|costs?|fees?)",
        r"versand(?:kosten)?",
        r"frais\s*(?:de\s*)?(?:port|livraison|exp[ée]dition)",
        r"(?:gastos\s*de\s*)?env[íi]o",
        r"(?:spese\s*di\s*)?spedizione",
    ),
    "rounding": _labels(
        r"rounding(?:\s*(?:adjustment|off))?",
        r"round[\s\-]*off",
        r"rundung(?:sdifferenz)?",
        r"arrondi",
        r"redondeo",
        r"arrotondamento",
    ),
    "deposit": _labels(
        r"deposit",
        r"advance(?:\s*(?:paid|payment))?",
        r"prepayment",
        r"amount\s*paid",
        r"payment\s*received",
        r"anzahlung",
        r"acompte",
        r"anticipo",
    ),
    # Indian GST is printed as two or three separate lines instead of one tax line.
    "cgst": _labels(r"cgst(?:\s*amount)?" + _RATE),
    "sgst": _labels(r"(?:sgst|utgst)(?:\s*amount)?" + _RATE),
    "igst": _labels(r"igst(?:\s*amount)?" + _RATE),
}
ALL_LABEL_PATTERNS = [label for labels in FIELD_LABELS.values() for label in labels]

_INVOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_\-.]*\d[A-Za-z0-9/_\-.]*$|^\d+$")
_GLUED_LABEL = re.compile(r"^([A-Za-z][A-Za-z.]*[:#])([^\s:/].*)$")
_ADDRESS_START = re.compile(r"^\d")
_CURRENCY_CODE = re.compile(r"^(USD|EUR|GBP|INR|JPY|CAD|AUD|CHF)$", re.IGNORECASE)
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
_DOC_WORDS = re.compile(
    r"^(tax\s+)?(invoice|bill|receipt|statement|rechnung|facture|factura|fattura|factuur|faktura|fatura|"
    r"nota\s+fiscal|credit\s+note|credit\s+memo|gutschrift|avoir)$",
    re.IGNORECASE,
)
# A line shaped "Label: value" is a detail of the document, never a company name.
_KEY_VALUE_LINE = re.compile(r"^[^\W\d_][\w .#/'’-]{0,30}:\s*\S")
_CREDIT_NOTE = re.compile(
    r"\b(credit\s*(?:note|memo)|gutschrift|avoir|nota\s*de\s*cr[ée]dito|nota\s*di\s*credito)\b", re.IGNORECASE
)
_SUMMARY_ROW = re.compile(
    r"\b(sub[\s\-]*total|total|tax|vat|gst|cgst|sgst|igst|utgst|balance|amount due|discount|shipping|freight|"
    r"handling|postage|rebate|zwischensumme|summe|mwst|ust|versand\w*|rabatt|gesamt\w*|sous[\s\-]*total|tva|"
    r"remise|iva|descuento|env[íi]o|imposta|sconto|spedizione|totale|imponibile|rounding|round[\s\-]*off|"
    r"deposit|advance|prepayment|amount paid|rundung\w*|anzahlung|arrondi|acompte|redondeo|anticipo|arrotondamento)\b",
    re.IGNORECASE,
)

_COLUMN_HEADERS = {
    "description": re.compile(
        r"\b(description|item|product|service|particulars|details|beschreibung|bezeichnung|artikel|position|"
        r"d[ée]signation|libell[ée]|descripci[óo]n|concepto|descrizione)\b",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        r"\b(qty|quantity|units?|hours?|hrs|menge|anzahl|quantit[ée]|qt[ée]|cantidad|cant|quantit[àa])\b",
        re.IGNORECASE,
    ),
    "unit_price": re.compile(
        r"\b(unit\s*price|unit\s*cost|price|rate|einzelpreis|preis|prix(?:\s*unitaire)?|precio(?:\s*unitario)?|"
        r"prezzo(?:\s*unitario)?)\b",
        re.IGNORECASE,
    ),
    "amount": re.compile(
        r"\b(amount|line\s*total|total|net|ext\w*|betrag|gesamt\w*|montant|importe|importo)\b", re.IGNORECASE
    ),
}


def _split_label_from_value(word: Word) -> list[Word]:
    """OCR can read 'Invoice No: 123' with no space, as 'No:123'. This splits such a word in two."""
    match = _GLUED_LABEL.match(word.text)
    if not match:
        return [word]
    left, right = match.group(1), match.group(2)
    x0, y0, x1, y1 = word.bbox
    split_at = x0 + (x1 - x0) * len(left) / len(word.text)
    return [
        word.model_copy(update={"text": left, "bbox": (x0, y0, split_at, y1)}),
        word.model_copy(update={"text": right, "bbox": (split_at, y0, x1, y1)}),
    ]


def _page_lines(page: Page) -> list[list[Word]]:
    """All readable words on a page as visual lines, skipping charts and pictures."""
    words: list[Word] = list(page.flagged_words)
    for region in page.regions:
        if region.type not in (RegionType.CHART, RegionType.IMAGE):
            words.extend(region.words)
    return group_into_lines([part for word in words for part in _split_label_from_value(word)])


def _split_at_gaps(line: list[Word]) -> list[list[Word]]:
    """Splits a line into separate pieces wherever there is a wide gap, such as between two columns."""
    if not line:
        return []
    pieces = [[line[0]]]
    for previous, word in zip(line, line[1:], strict=False):
        height = max(1.0, word.bbox[3] - word.bbox[1])
        if word.bbox[0] - previous.bbox[2] > 2.2 * height:
            pieces.append([word])
        else:
            pieces[-1].append(word)
    return pieces


def _match_label(words: list[Word], labels: list[Label]) -> tuple[int, Label] | None:
    """Finds the longest label at the start of these words. Returns how many words it covers."""
    for count in range(min(5, len(words)), 0, -1):
        text = " ".join(word.text for word in words[:count])
        for label in labels:
            if label.pattern.match(text):
                return count, label
    return None


@dataclass
class Candidate:
    words: list[Word]
    page: int
    rank: int

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


def _is_any_label(words: list[Word]) -> bool:
    return _match_label(words, ALL_LABEL_PATTERNS) is not None


def _is_exactly_a_label(words: list[Word]) -> bool:
    text = " ".join(word.text for word in words)
    return any(label.pattern.match(text) for label in ALL_LABEL_PATTERNS)


def _value_after_label(remaining: list[Word]) -> list[Word]:
    """The value runs until the next label on the same line, if one starts there."""
    value: list[Word] = []
    for index, word in enumerate(remaining):
        if index > 0 and _is_any_label(remaining[index:]):
            break
        value.append(word)
    return value


def _value_below(line_index: int, label_box: BBox, lines: list[list[Word]]) -> list[Word]:
    """Looks in the next line for words directly under the label."""
    if line_index + 1 >= len(lines):
        return []
    height = max(1.0, label_box[3] - label_box[1])
    below = lines[line_index + 1]
    if below[0].bbox[1] - label_box[3] > 2.5 * height:
        return []
    for piece in _split_at_gaps(below):
        box = union_box(word.bbox for word in piece)
        if box[0] < label_box[2] and box[2] > label_box[0] and not _is_any_label(piece):
            return piece
    return []


def _candidates(document: ExtractedDocument, labels: list[Label]) -> list[Candidate]:
    found: list[Candidate] = []
    for page in document.pages:
        lines = _page_lines(page)
        for line_index, line in enumerate(lines):
            pieces = _split_at_gaps(line)
            for index, piece in enumerate(pieces):
                for start in range(len(piece)):
                    match = _match_label(piece[start:], labels)
                    if match is None:
                        continue
                    # A label must begin the piece or follow another value, not sit inside a sentence.
                    if start > 0 and len(piece) > start + match[0] + 3:
                        continue
                    # "Date" in "Due Date" isn't a separate label, because "Due Date" is a longer one.
                    if any(_is_exactly_a_label(piece[earlier : start + match[0]]) for earlier in range(start)):
                        continue
                    count, label = match
                    remaining = piece[start + count :]
                    value = _value_after_label(remaining)
                    # The value can be in the next piece along the line, as in a table row.
                    if not value and index + 1 < len(pieces) and not _is_any_label(pieces[index + 1]):
                        value = pieces[index + 1]
                    if not value:
                        label_box = union_box(word.bbox for word in piece[start : start + count])
                        value = _value_below(line_index, label_box, lines)
                    if value:
                        found.append(Candidate(value, page.number, label.rank))
                    break
    return found


def _source_of(words: list[Word]) -> str:
    sources = {word.source for word in words}
    if sources == {WordSource.TEXT_LAYER}:
        return "text_layer"
    if sources == {WordSource.OCR}:
        return "ocr"
    return "mixed"


def _ocr_confidence(words: list[Word]) -> float | None:
    values = [word.confidence for word in words if word.confidence is not None]
    return min(values) if values else None


def _field(candidate: Candidate, value: Any, alternatives: list[Any] | None = None) -> FieldValue:
    return FieldValue(
        value=value,
        raw=candidate.text,
        page=candidate.page,
        position=union_box(word.bbox for word in candidate.words),
        source=_source_of(candidate.words),
        ocr_confidence=_ocr_confidence(candidate.words),
        alternatives=alternatives or [],
    )


def _best(candidates: list[Candidate]) -> list[Candidate]:
    if not candidates:
        return []
    best_rank = min(candidate.rank for candidate in candidates)
    return [candidate for candidate in candidates if candidate.rank == best_rank]


def _pick(parsed: list[tuple[Candidate, Any]]) -> FieldValue:
    """Takes the first value, and records any different values so the checks can flag the conflict."""
    if not parsed:
        return FieldValue()
    first_candidate, first_value = parsed[0]
    others = [value for _, value in parsed[1:] if value != first_value]
    return _field(first_candidate, first_value, others)


def find_invoice_number(document: ExtractedDocument) -> FieldValue:
    parsed = []
    for candidate in _best(_candidates(document, FIELD_LABELS["invoice_number"])):
        token = candidate.words[0].text.strip(":#.,")
        if _INVOICE_ID.match(token):
            parsed.append((Candidate(candidate.words[:1], candidate.page, candidate.rank), token))
    return _pick(parsed)


def find_date(document: ExtractedDocument, name: str) -> FieldValue:
    parsed = []
    ambiguous = False
    for candidate in _best(_candidates(document, FIELD_LABELS[name])):
        # A date is at most five words, for example "5 de marzo de 2026". Longer text can't parse as one date.
        for length in range(min(5, len(candidate.words)), 0, -1):
            part = candidate.words[:length]
            value, unclear = parse_date(" ".join(word.text for word in part))
            if value is not None:
                ambiguous = ambiguous or unclear
                parsed.append((Candidate(part, candidate.page, candidate.rank), value.isoformat()))
                break
    result = _pick(parsed)
    if result.value is None and _best(_candidates(document, FIELD_LABELS[name])):
        first = _best(_candidates(document, FIELD_LABELS[name]))[0]
        result = _field(first, None)
    if ambiguous:
        result.rule_notes.append("the date could be read as day-first or month-first")
    return result


def find_money(document: ExtractedDocument, name: str) -> FieldValue:
    parsed: list[tuple[Candidate, Decimal]] = []
    for candidate in _best(_candidates(document, FIELD_LABELS[name])):
        for index in range(len(candidate.words)):
            value = parse_money(candidate.words[index].text)
            if value is None and index + 1 < len(candidate.words):
                value = parse_money(" ".join(w.text for w in candidate.words[index : index + 2]))
            if value is not None:
                parsed.append((Candidate(candidate.words[index : index + 1], candidate.page, candidate.rank), value))
                break
    return _pick(parsed)


def find_text(document: ExtractedDocument, name: str) -> FieldValue:
    parsed = [(candidate, candidate.text.strip()) for candidate in _best(_candidates(document, FIELD_LABELS[name]))]
    return _pick(parsed)


def find_vendor(document: ExtractedDocument) -> FieldValue:
    """A vendor label wins. Otherwise uses the first text at the top of page 1 that isn't 'Invoice'.

    A heading in the top third of the page is decent evidence, so it only lowers the score. Any
    other position is weak evidence, so that value always goes to a person.
    """
    labelled = find_text(document, "vendor_name")
    if labelled.value:
        return labelled
    if not document.pages:
        return FieldValue()
    page = document.pages[0]
    heading_boxes = [region.bbox for region in page.regions if region.type is RegionType.HEADING]
    for line in _page_lines(page):
        if line[0].bbox[1] > page.height / 3:
            break
        first_piece = _split_at_gaps(line)[0]
        text = " ".join(word.text for word in first_piece)
        if (
            _DOC_WORDS.match(text)
            or _is_any_label(first_piece)
            or _ADDRESS_START.match(text)
            or _KEY_VALUE_LINE.match(text)
        ):
            continue
        result = _field(Candidate(first_piece, page.number, 0), text)
        in_heading = any(overlap_of(first_piece[0].bbox, box) >= 0.5 for box in heading_boxes)
        if in_heading:
            result.evidence_penalty = 0.03
        else:
            result.rule_notes.append("no vendor label was found, so this is taken from the top of the page")
        return result
    return FieldValue()


def find_currency(document: ExtractedDocument) -> FieldValue:
    for candidate in _best(_candidates(document, FIELD_LABELS["currency"])):
        token = candidate.words[0].text
        if _CURRENCY_CODE.match(token):
            return _field(Candidate(candidate.words[:1], candidate.page, 0), token.upper())
    for page in document.pages:
        for line in _page_lines(page):
            for word in line:
                for symbol, code in _CURRENCY_SYMBOLS.items():
                    if symbol in word.text and parse_money(word.text) is not None:
                        return _field(Candidate([word], page.number, 0), code)
    return FieldValue()


def _map_columns(rows: list[list[str]]) -> tuple[dict[str, int], int] | None:
    """Reads column names from the first rows. Headers can span two rows, so both are combined."""
    n_cols = max((len(row) for row in rows), default=0)
    for header_rows in (1, 2):
        if len(rows) <= header_rows:
            continue
        combined = [" ".join(rows[r][c] for r in range(header_rows)).strip() for c in range(n_cols)]
        columns: dict[str, int] = {}
        # The most specific headers choose first, so "Unit Price" isn't taken as the amount column.
        for name in ("unit_price", "quantity", "description", "amount"):
            for col, text in enumerate(combined):
                if col not in columns.values() and _COLUMN_HEADERS[name].search(text):
                    columns[name] = col
                    break
        if "description" in columns and "amount" in columns:
            return columns, header_rows
    return None


def _is_totals_box(table: JoinedTable) -> bool:
    """A small table that only holds Subtotal, Tax and Total rows, as most invoices print beside their totals."""
    rows = [row for row in table.rows if any(cell.strip() for cell in row)]
    if not rows:
        return False
    return all(_SUMMARY_ROW.search(next((cell for cell in row if cell.strip()), "")) for row in rows)


def has_possible_item_table(document: ExtractedDocument) -> bool:
    """True if any table could be a list of items, even one whose headers weren't recognised.

    A totals box doesn't count. Any other table might be an item list that was missed, so an invoice with one is
    never treated as having no items.
    """
    return any(not _is_totals_box(table) for table in document.tables)


def find_line_items(document: ExtractedDocument) -> FieldValue:
    """Reads items from the table whose headers look like an item list. Summary rows are left out."""
    for table in document.tables:
        mapped = _map_columns(table.rows)
        if mapped is None:
            continue
        columns, header_rows = mapped
        return _read_items(table, columns, header_rows)
    return FieldValue()


def _read_items(table: JoinedTable, columns: dict[str, int], header_rows: int) -> FieldValue:
    items: list[dict[str, Any]] = []
    boxes: list[BBox] = []
    confidences: list[float] = []
    pages: list[int] = []
    unreadable = 0
    for index, row in enumerate(table.rows):
        if index < header_rows:
            continue
        text = " ".join(row)
        description = row[columns["description"]].strip()
        amount_text = row[columns["amount"]].strip()
        if not text.strip():
            continue
        if _SUMMARY_ROW.search(description) or (not description and _SUMMARY_ROW.search(text)):
            continue
        if not amount_text and items and description:
            # A description that wrapped onto its own row belongs to the item above.
            items[-1]["description"] = f"{items[-1]['description']} {description}"
            continue
        amount = parse_money(amount_text)
        quantity = parse_money(row[columns["quantity"]]) if "quantity" in columns else None
        unit_price = parse_money(row[columns["unit_price"]]) if "unit_price" in columns else None
        if amount is None:
            unreadable += 1
        items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "amount": amount,
            }
        )
        if index < len(table.row_sources):
            page, box = table.row_sources[index]
            pages.append(page)
            boxes.append(box)
        if index < len(table.row_min_confidences):
            confidence = table.row_min_confidences[index]
            if confidence is not None:
                confidences.append(confidence)
    if not items:
        return FieldValue()
    result = FieldValue(
        value=items,
        raw=table.id,
        page=pages[0] if pages else table.pages[0],
        position=boxes[0] if boxes else None,
        source="ocr" if confidences else "text_layer",
        ocr_confidence=min(confidences) if confidences else None,
    )
    if unreadable:
        result.rule_notes.append(f"{unreadable} line items have an unreadable amount")
    return result


def detect_document_type(document: ExtractedDocument) -> str:
    """ "credit_note" if the top of the first page says so, otherwise "invoice".

    A credit note repeats an invoice's layout with amounts that are owed back, so they may print as negatives.
    """
    if not document.pages:
        return "invoice"
    page = document.pages[0]
    for line in _page_lines(page):
        if line[0].bbox[1] > page.height / 2.5:
            break
        if _CREDIT_NOTE.search(" ".join(word.text for word in line)):
            return "credit_note"
    return "invoice"


def extract_fields(document: ExtractedDocument) -> dict[str, FieldValue]:
    return {
        "invoice_number": find_invoice_number(document),
        "invoice_date": find_date(document, "invoice_date"),
        "due_date": find_date(document, "due_date"),
        "vendor_name": find_vendor(document),
        "bill_to": find_text(document, "bill_to"),
        "currency": find_currency(document),
        "line_items": find_line_items(document),
        "subtotal": find_money(document, "subtotal"),
        "tax": find_money(document, "tax"),
        "grand_total": find_money(document, "grand_total"),
        "discount": find_money(document, "discount"),
        "shipping": find_money(document, "shipping"),
        "rounding": find_money(document, "rounding"),
        "deposit": find_money(document, "deposit"),
        "cgst": find_money(document, "cgst"),
        "sgst": find_money(document, "sgst"),
        "igst": find_money(document, "igst"),
    }
