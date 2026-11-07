"""Runs the whole pipeline on one PDF and produces the final result."""

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .engines import Engines
from .fields import checks, recovery, rules, visible
from .models import DocumentResult
from .pipeline.extract import extract_document
from .pipeline.intake import inspect_pdf


def process_pdf(
    path: Path,
    settings: Settings,
    engines: Engines,
    output_dir: Path | None = None,
    llm: Any = None,
    llm_budget_usd: float | None = None,
) -> DocumentResult:
    started = time.perf_counter()
    info = inspect_pdf(path, settings)  # Step 1
    document = extract_document(path, settings, engines, output_dir)  # Steps 2 to 12
    fields = rules.extract_fields(document)  # Step 13
    # A digital page's text layer can differ from what the page shows, so the key values are compared with it.
    if settings.visible_check_enabled:
        visible.verify_visible_values(
            fields, path, {page.number: page.type for page in document.pages}, settings, engines
        )
    checks.run_checks(fields, document, settings)  # Step 14
    # Rule notes like "ambiguous date" were turned into failed checks above.
    checks.score_and_decide(fields, document, settings)  # Step 15
    if settings.recovery_enabled:
        # Step 15b: fields that failed a check on a scan are read again in other ways, and kept only if all checks pass.
        recovery.second_pass(fields, document, path, settings, engines)

    usage = None
    if llm is not None:
        usage = llm.resolve_failed_fields(fields, document, path, llm_budget_usd)  # Step 16

    result = DocumentResult(
        fingerprint=info.fingerprint,
        pipeline_version=__version__,
        status="needs_review" if any(field.needs_review for field in fields.values()) else "accepted",
        document_type=rules.detect_document_type(document),
        page_count=info.page_count,
        fields=fields,
        pages=document.pages,
        tables=document.tables,
        seconds=round(time.perf_counter() - started, 2),
    )
    if usage is not None:
        result.llm = usage
    return result


def result_to_json(result: DocumentResult) -> str:
    return json.dumps(result_to_dict(result), indent=2, ensure_ascii=False)


def result_to_dict(result: DocumentResult) -> dict[str, Any]:
    """Money becomes a string like "250.00", so JSON never loses an exact amount."""
    plain: dict[str, Any] = _plain(result.model_dump(mode="python"))
    return plain


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and value.__class__.__module__.startswith("pdfpipe"):
        return value.value
    return value
