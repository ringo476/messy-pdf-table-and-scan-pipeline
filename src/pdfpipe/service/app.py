"""The web app and API. Every job route is limited to the logged-in user's own jobs."""

import copy
import logging
import shutil
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from importlib import resources
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Any, cast

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from .. import __version__
from ..config import Settings
from ..fields.checks import INVOICE_ID, expected_totals
from ..fields.llm import ClaudeResolver
from ..fields.values import parse_date, parse_money
from ..pipeline.intake import RejectedFile
from . import ask, auth, db
from .safe_intake import inspect_in_process
from .worker import copy_reused_files, job_dir

logger = logging.getLogger("pdfpipe.app")

_LOGIN_ATTEMPTS = 8
_LOGIN_WINDOW_SECONDS = 300


class Credentials(BaseModel):
    username: str
    password: str


class FieldEdit(BaseModel):
    name: str
    value: Any = None


class Question(BaseModel):
    question: str


class ReviewRequest(BaseModel):
    edits: list[FieldEdit] = []
    approve: list[str] = []
    finish: bool = False


class LoginLimiter:
    """Slows password guessing by limiting failed logins per username and address."""

    def __init__(self) -> None:
        self.failures: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def blocked(self, key: str) -> bool:
        with self.lock:
            attempts = self.failures[key]
            while attempts and monotonic() - attempts[0] > _LOGIN_WINDOW_SECONDS:
                attempts.popleft()
            return len(attempts) >= _LOGIN_ATTEMPTS

    def record_failure(self, key: str) -> None:
        with self.lock:
            self.failures[key].append(monotonic())

    def clear(self, key: str) -> None:
        with self.lock:
            self.failures.pop(key, None)


_MONEY_FIELDS = ("subtotal", "tax", "grand_total", "discount", "shipping", "cgst", "sgst", "igst")
_DATE_FIELDS = ("invoice_date", "due_date")
_REQUIRED_FIELDS = ("invoice_number", "invoice_date", "vendor_name", "grand_total")


def clean_edit(name: str, value: Any, credit_note: bool = False) -> Any:
    """Checks a person's typed value the same way the automatic one was checked, and returns it in the stored form."""
    text = "" if value is None else str(value).strip()
    if not text:
        if name in _REQUIRED_FIELDS:
            raise HTTPException(status_code=422, detail=f"{name} can't be empty.")
        return None
    if name == "line_items":
        raise HTTPException(
            status_code=422, detail="Line items can't be edited here. Approve or correct the other fields."
        )
    if name in _MONEY_FIELDS:
        amount = parse_money(text)
        # A discount is often printed as a negative, and credit notes print negatives throughout.
        if amount is None or (amount < 0 and name != "discount" and not credit_note):
            raise HTTPException(status_code=422, detail=f"{name} must be a clear, non-negative amount like 1250.00.")
        return format(amount, "f")
    if name in _DATE_FIELDS:
        day, ambiguous = parse_date(text)
        if day is None or ambiguous or not 2000 <= day.year <= 2100:
            raise HTTPException(status_code=422, detail=f"{name} must be a clear date like 2026-03-25.")
        return day.isoformat()
    if name == "invoice_number" and not INVOICE_ID.match(text):
        raise HTTPException(status_code=422, detail="Invoice numbers use letters, digits and - / _ .")
    if name == "vendor_name" and not any(character.isalpha() for character in text):
        raise HTTPException(status_code=422, detail="A vendor name must contain letters.")
    return text


def apply_review(
    machine: dict[str, Any], current: dict[str, Any] | None, request: ReviewRequest, tolerance: float = 0.01
) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    credit_note = machine.get("document_type") == "credit_note"
    """Applies a person's edits on top of the machine's fields. The machine's output stays unchanged.

    Typed values go through the same format and total checks as automatic ones. A person who believes the
    document itself is wrong can approve the printed values instead of editing them.
    """
    fields: dict[str, Any] = copy.deepcopy(current["fields"]) if current else copy.deepcopy(machine["fields"])
    changes: list[tuple[str, Any, Any]] = []
    for edit in request.edits:
        if edit.name not in fields:
            raise HTTPException(status_code=422, detail=f"Unknown field: {edit.name}")
        field = fields[edit.name]
        cleaned = clean_edit(edit.name, edit.value, credit_note)
        changes.append((edit.name, field.get("value"), cleaned))
        field.update(value=cleaned, source="person", needs_review=False, confidence=1.0, review_reasons=[])
    if any(edit.name in _MONEY_FIELDS for edit in request.edits) and not credit_note:
        _check_edited_totals(fields, Decimal(str(tolerance)))
    for name in request.approve:
        if name not in fields:
            raise HTTPException(status_code=422, detail=f"Unknown field: {name}")
        field = fields[name]
        changes.append((name, field.get("value"), field.get("value")))
        field.update(needs_review=False, review_reasons=[], approved_by_person=True)
    return {"fields": fields}, changes


def _check_edited_totals(fields: dict[str, Any], tolerance: Decimal) -> None:
    def amount(name: str) -> Decimal | None:
        value = fields.get(name, {}).get("value")
        return parse_money(str(value)) if value not in (None, "") else None

    subtotal, total = amount("subtotal"), amount("grand_total")
    if subtotal is None or total is None:
        return
    candidates = expected_totals(
        subtotal,
        amount("tax"),
        amount("cgst"),
        amount("sgst"),
        amount("igst"),
        amount("shipping"),
        amount("discount"),
    )
    if not any(abs(candidate - total) <= tolerance for candidate in candidates):
        raise HTTPException(
            status_code=422,
            detail=f"The subtotal plus tax and other charges is {candidates[0]}, but the grand total is {total}. "
            "Correct the values, or approve the printed ones if the document itself is wrong.",
        )


def create_app(settings: Settings | None = None, claude_client: Any = None) -> FastAPI:
    """claude_client is only for tests. With Claude turned on, the real client is created when it is first needed."""
    settings = settings or Settings()
    uploads_dir = settings.data_dir / "uploads"
    pool = ConnectionPool(
        settings.database_url, min_size=1, max_size=10, open=False, kwargs={"autocommit": True, "row_factory": dict_row}
    )
    limiter = LoginLimiter()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        uploads_dir.mkdir(parents=True, exist_ok=True)
        pool.open(wait=True, timeout=60)
        with pool.connection() as connection:
            db.apply_schema(cast(db.Connection, connection))
        yield
        pool.close()

    app = FastAPI(title="Messy PDF Pipeline", version=__version__, lifespan=lifespan)

    def get_connection() -> Iterator[db.Connection]:
        with pool.connection() as connection:
            yield cast(db.Connection, connection)

    Conn = Annotated[db.Connection, Depends(get_connection)]

    def current_user(connection: Conn, pdfpipe_session: Annotated[str | None, Cookie()] = None) -> dict[str, Any]:
        user = auth.user_for_session(connection, pdfpipe_session)
        if user is None:
            raise HTTPException(status_code=401, detail="Please log in.")
        return user

    User = Annotated[dict[str, Any], Depends(current_user)]

    def own_job(job_id: uuid.UUID, connection: Conn, user: User) -> dict[str, Any]:
        job = db.get_job(connection, job_id, user["id"])
        if job is None:
            # The same reply whether the job doesn't exist or belongs to someone else.
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    Job = Annotated[dict[str, Any], Depends(own_job)]

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
        )
        return response

    @app.get("/health")
    def health(connection: Conn) -> dict[str, Any]:
        row = connection.execute(
            """SELECT count(*) FILTER (WHERE status = 'queued') AS queued,
                      count(*) FILTER (WHERE status = 'running') AS running
               FROM jobs"""
        ).fetchone()
        return {"status": "ok", "version": __version__, **(row or {})}

    @app.post("/api/login")
    def login(body: Credentials, request: Request, response: Response, connection: Conn) -> dict[str, str]:
        address = request.client.host if request.client else "unknown"
        key = f"{body.username}|{address}"
        if limiter.blocked(key):
            raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")
        user_id = auth.check_login(connection, body.username, body.password)
        if user_id is None:
            limiter.record_failure(key)
            raise HTTPException(status_code=401, detail="Wrong username or password.")
        limiter.clear(key)
        token = auth.start_session(connection, user_id, settings.session_hours)
        response.set_cookie(
            auth.SESSION_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=settings.cookie_secure,
            max_age=settings.session_hours * 3600,
        )
        return {"username": body.username}

    @app.post("/api/logout")
    def logout(
        response: Response, connection: Conn, pdfpipe_session: Annotated[str | None, Cookie()] = None
    ) -> dict[str, bool]:
        auth.end_session(connection, pdfpipe_session)
        response.delete_cookie(auth.SESSION_COOKIE)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: User) -> dict[str, Any]:
        return {"username": user["username"], "can_ask": settings.llm_enabled or claude_client is not None}

    @app.get("/api/invoices")
    def invoice_search(
        connection: Conn,
        user: User,
        vendor: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict[str, Any]:
        """The user's own accepted and reviewed invoices, with totals worked out by code."""
        rows = db.search_invoices(connection, user["id"], vendor, date_from, date_to)
        totals = ask.total_by_currency(rows)
        filters = {"vendor": vendor or None, "date_from": date_from, "date_to": date_to}
        return db.to_jsonable(
            {"invoices": rows, "totals": totals, "summary": ask.describe(filters, rows, totals), "filters": filters}
        )

    @app.post("/api/ask")
    def ask_about_invoices(body: Question, connection: Conn, user: User) -> dict[str, Any]:
        client = claude_client
        if client is None and settings.llm_enabled:
            client = ClaudeResolver(settings).client
        if client is None:
            raise HTTPException(status_code=503, detail="Questions in plain words need Claude to be turned on.")
        filters = ask.parse_question(client, settings, body.question, date.today())
        if filters is None:
            raise HTTPException(
                status_code=422, detail="I couldn't understand that question. Use the vendor and date boxes instead."
            )
        rows = db.search_invoices(connection, user["id"], filters["vendor"], filters["date_from"], filters["date_to"])
        totals = ask.total_by_currency(rows)
        return db.to_jsonable(
            {"invoices": rows, "totals": totals, "summary": ask.describe(filters, rows, totals), "filters": filters}
        )

    @app.post("/api/jobs", status_code=202)
    def upload(file: UploadFile, connection: Conn, user: User) -> dict[str, Any]:
        if db.active_job_count(connection, user["id"]) >= settings.max_active_jobs_per_user:
            raise HTTPException(
                status_code=429, detail="You have too many documents waiting. Please wait for some to finish."
            )

        # The file goes to a temporary name first, and the size limit is enforced while reading it.
        job_id = uuid.uuid4()
        temporary = uploads_dir / f"{job_id}.tmp"
        limit = settings.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with temporary.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            status_code=413, detail=f"The file is larger than the {settings.max_upload_mb} MB limit."
                        )
                    target.write(chunk)
            try:
                info = inspect_in_process(temporary, settings)
            except RejectedFile as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

            # The name is display text only. It never becomes part of a path on disk.
            filename = Path(file.filename or "document.pdf").name[:200]
            folder = job_dir(settings, job_id)
            folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), folder / "input.pdf")
        finally:
            temporary.unlink(missing_ok=True)

        db.create_job(connection, job_id, user["id"], filename, info.fingerprint, info.size_bytes, info.page_count)

        # The same file already processed by this pipeline version doesn't need processing again.
        reusable = db.find_reusable(connection, user["id"], info.fingerprint, __version__)
        if reusable is not None:
            machine = reusable["result"]["machine"]
            copy_reused_files(settings, reusable["id"], job_id)
            db.finish_with_reused_result(connection, job_id, reusable["id"], machine)
            return {"id": str(job_id), "status": machine["status"], "reused": True}
        return {"id": str(job_id), "status": "queued", "reused": False}

    @app.get("/api/jobs")
    def jobs(connection: Conn, user: User) -> list[dict[str, Any]]:
        return [db.to_jsonable(row) for row in db.list_jobs(connection, user["id"])]

    @app.get("/api/jobs/{job_id}")
    def job_detail(job: Job, connection: Conn, user: User) -> JSONResponse:
        data = db.to_jsonable({key: value for key, value in job.items() if key != "user_id"})
        data["history"] = [db.to_jsonable(row) for row in db.review_history(connection, job["id"], user["id"])]
        return JSONResponse(data)

    @app.get("/api/jobs/{job_id}/result.json")
    def download_result(job: Job) -> JSONResponse:
        if not job["result"]:
            raise HTTPException(status_code=409, detail="This document isn't finished yet.")
        result = copy.deepcopy(job["result"]["machine"])
        reviewed = job["result"].get("reviewed")
        if reviewed:
            result["fields"] = reviewed["fields"]
        return JSONResponse(result, headers={"Content-Disposition": f'attachment; filename="{job["id"]}.json"'})

    @app.get("/api/jobs/{job_id}/charts/{name}")
    def chart(name: str, job: Job) -> FileResponse:
        # A chart name is exactly one path part, so it can't reach outside the job's folder.
        if name != Path(name).name or not name.endswith(".png"):
            raise HTTPException(status_code=404, detail="Chart not found.")
        path = job_dir(settings, job["id"]) / "charts" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Chart not found.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/jobs/{job_id}/review")
    def review(job: Job, body: ReviewRequest, connection: Conn, user: User) -> dict[str, Any]:
        if not job["result"]:
            raise HTTPException(status_code=409, detail="This document isn't finished yet.")
        reviewed, changes = apply_review(
            job["result"]["machine"], job["result"].get("reviewed"), body, settings.money_tolerance
        )
        still_open = [name for name, field in reviewed["fields"].items() if field.get("needs_review")]
        if body.finish and still_open:
            raise HTTPException(status_code=409, detail=f"These fields still need a decision: {', '.join(still_open)}")
        if not db.save_review(connection, job["id"], user["id"], reviewed, changes, body.finish):
            raise HTTPException(status_code=409, detail="This document can't be reviewed right now.")
        if body.finish:
            # What a person confirmed is the most trustworthy record of this invoice.
            db.record_invoice(
                connection, job["id"], user["id"], "reviewed", job["result"]["machine"], reviewed["fields"]
            )
        return {"still_needs_review": still_open, "finished": body.finish}

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete(job: Job, connection: Conn, user: User) -> Response:
        if not db.delete_job(connection, job["id"], user["id"]):
            raise HTTPException(status_code=409, detail="This document is still being processed.")
        shutil.rmtree(job_dir(settings, job["id"]), ignore_errors=True)
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("pdfpipe.service").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/app.js", include_in_schema=False)
    def script() -> Response:
        text = resources.files("pdfpipe.service").joinpath("static/app.js").read_text(encoding="utf-8")
        return Response(text, media_type="text/javascript")

    return app


def app_factory() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
