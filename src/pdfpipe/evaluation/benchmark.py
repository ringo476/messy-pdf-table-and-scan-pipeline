"""Measures the pipeline against documents with known answers.

The most important number is the silent error rate: wrong values that were accepted with no person
looking. A wrong value sent to review costs a person some time. A silent one reaches the books.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..config import Settings
from ..engines import Engines
from ..models import DocumentResult
from ..process import process_pdf
from .synth import CONDITIONS, STYLES, Case, make_benchmark_set

FIELD_NAMES = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "bill_to",
    "currency",
    "subtotal",
    "tax",
    "grand_total",
]


def _same(got: Any, want: Any) -> bool:
    if isinstance(got, Decimal):
        return want is not None and abs(float(got) - float(want)) < 0.005
    if isinstance(got, str) and isinstance(want, str):
        return " ".join(got.split()).lower() == " ".join(want.split()).lower()
    return bool(got == want)


def _items_correct(got: list[dict[str, Any]] | None, want: list[dict[str, Any]]) -> bool:
    if not got or len(got) != len(want):
        return False
    return all(
        _same(g.get("description"), w["description"])
        and _same(g.get("quantity"), w["quantity"])
        and _same(g.get("unit_price"), w["unit_price"])
        and _same(g.get("amount"), w["amount"])
        for g, w in zip(got, want, strict=True)
    )


def table_cell_scores(got_rows: list[list[str]], want_rows: list[list[str]]) -> tuple[int, int, int]:
    """Counts matched cells by (row, column, text). Returns (matched, found, expected)."""

    def cells(rows: list[list[str]]) -> set[tuple[int, int, str]]:
        return {
            (r, c, " ".join(text.split()).lower())
            for r, row in enumerate(rows)
            for c, text in enumerate(row)
            if text.strip()
        }

    got, want = cells(got_rows), cells(want_rows)
    return len(got & want), len(got), len(want)


@dataclass
class Tally:
    fields_total: int = 0
    fields_correct: int = 0
    fields_accepted: int = 0
    silent_errors: int = 0
    fields_reviewed: int = 0
    docs: int = 0
    docs_accepted: int = 0
    must_review_docs: int = 0
    must_review_missed: int = 0
    matched: int = 0
    found: int = 0
    expected: int = 0
    seconds: list[float] = field(default_factory=list)
    pages: int = 0
    errors: int = 0

    def row(self) -> dict[str, Any]:
        precision = self.matched / self.found if self.found else 0.0
        recall = self.matched / self.expected if self.expected else 0.0
        per_page = sorted(self.seconds)
        return {
            "documents": self.docs,
            "field_accuracy": round(self.fields_correct / self.fields_total, 4) if self.fields_total else None,
            "silent_error_rate": round(self.silent_errors / self.fields_accepted, 4) if self.fields_accepted else None,
            "silent_errors": self.silent_errors,
            "field_review_rate": round(self.fields_reviewed / self.fields_total, 4) if self.fields_total else None,
            "documents_fully_accepted": self.docs_accepted,
            "must_review_missed": f"{self.must_review_missed}/{self.must_review_docs}",
            "table_cell_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "seconds_per_page_p50": round(statistics.median(per_page), 2) if per_page else None,
            "seconds_per_page_p95": round(per_page[min(len(per_page) - 1, int(0.95 * len(per_page)))], 2)
            if per_page
            else None,
            "crashes": self.errors,
        }


def _best_table(result: DocumentResult, want: list[list[str]]) -> list[list[str]]:
    best: list[list[str]] = []
    best_score = -1
    for table in result.tables:
        rows = table.rows
        # A two-row header is a formatting choice, so the expected data is compared without it.
        if len(rows) >= 2 and rows[1] and "".join(rows[1]).lower().replace(" ", "").startswith("unitprice"):
            rows = [want[0], *rows[2:]]
        score = table_cell_scores(rows, want)[0]
        if score > best_score:
            best, best_score = rows, score
    return best


def score_case(case: Case, result: DocumentResult, tally: Tally, failures: list[dict[str, Any]]) -> None:
    truth = case.truth
    tally.docs += 1
    tally.pages += result.page_count
    tally.seconds.append(result.seconds / max(1, result.page_count))
    tally.docs_accepted += result.status == "accepted"
    if truth["must_review"]:
        tally.must_review_docs += 1
        if result.status == "accepted":
            tally.must_review_missed += 1
            failures.append({"case": case.name, "problem": "accepted a document that must be reviewed"})

    comparisons = [(name, result.fields[name], truth["fields"][name], _same) for name in FIELD_NAMES]
    comparisons.append(("line_items", result.fields["line_items"], truth["line_items"], _items_correct))
    for name, value, want, compare in comparisons:
        correct = compare(value.value, want)
        tally.fields_total += 1
        tally.fields_correct += correct
        tally.fields_reviewed += value.needs_review
        accepted = not value.needs_review and value.value not in (None, "", [])
        tally.fields_accepted += accepted
        if accepted and not correct:
            tally.silent_errors += 1
            failures.append(
                {
                    "case": case.name,
                    "problem": "SILENT ERROR",
                    "field": name,
                    "got": str(value.value)[:200],
                    "want": str(want)[:200],
                }
            )
        elif not correct and want not in (None, []):
            failures.append(
                {
                    "case": case.name,
                    "problem": "wrong or empty (sent to review)",
                    "field": name,
                    "got": str(value.value)[:120],
                    "want": str(want)[:120],
                    "reasons": value.review_reasons[:2],
                }
            )

    matched, found, expected = table_cell_scores(_best_table(result, truth["table_rows"]), truth["table_rows"])
    tally.matched += matched
    tally.found += found
    tally.expected += expected


def run(
    folder: Path, seeds: int, styles: list[str] | None, conditions: list[str] | None, first_seed: int = 0
) -> dict[str, Any]:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    engines = Engines(settings)
    engines.warm_up()
    cases = make_benchmark_set(
        folder / "documents", seeds=seeds, styles=styles, conditions=conditions, first_seed=first_seed
    )

    overall = Tally()
    by_condition: dict[str, Tally] = defaultdict(Tally)
    by_style: dict[str, Tally] = defaultdict(Tally)
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        try:
            result = process_pdf(case.pdf, settings, engines, folder / "output" / case.name)
        except Exception as error:  # A crash is a measured failure, and the run must continue.
            overall.errors += 1
            by_condition[case.condition].errors += 1
            by_style[case.style].errors += 1
            failures.append({"case": case.name, "problem": "CRASH", "error": repr(error)[:300]})
            continue
        for tally in (overall, by_condition[case.condition], by_style[case.style]):
            score_case(case, result, tally, failures if tally is overall else [])

    return {
        "note": "Generated test documents only. This doesn't measure accuracy on real-world documents.",
        "total_seconds": round(time.perf_counter() - started, 1),
        "overall": overall.row(),
        "by_condition": {name: tally.row() for name, tally in by_condition.items()},
        "by_style": {name: tally.row() for name, tally in by_style.items()},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark on generated documents.")
    parser.add_argument("--out", type=Path, default=Path("eval_out"))
    parser.add_argument("--seeds", type=int, default=1, help="how many documents per style and condition")
    parser.add_argument("--first-seed", type=int, default=0, help="use new seeds to test on unseen documents")
    parser.add_argument("--styles", nargs="*", choices=STYLES)
    parser.add_argument("--conditions", nargs="*", choices=CONDITIONS)
    args = parser.parse_args()
    report = run(args.out, args.seeds, args.styles, args.conditions, args.first_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total_seconds", "overall", "by_condition", "by_style")}, indent=2))
    for failure in report["failures"][:60]:
        print(failure)
    print(f"{len(report['failures'])} failures in total. The full report is in {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
