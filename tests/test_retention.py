"""The retention rule. It is off by default, and it never touches a document that still needs review."""

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pdfpipe.config import Settings
from pdfpipe.service import auth, db
from pdfpipe.service.paths import job_dir
from pdfpipe.service.retention import run_retention

DATABASE_URL = os.environ.get("PDFPIPE_TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DATABASE_URL, reason="PDFPIPE_TEST_DATABASE_URL isn't set")]


def settings_for(tmp_path: Path, **changes: Any) -> Settings:
    return Settings(_env_file=None, database_url=DATABASE_URL, data_dir=tmp_path, **changes)  # type: ignore[arg-type]


@pytest.fixture
def connection(tmp_path: Path) -> Any:
    with db.connect(settings_for(tmp_path)) as conn:
        db.apply_schema(conn)
        conn.execute("TRUNCATE users, sessions, jobs, review_log CASCADE")
        yield conn


def add_job(connection: Any, settings: Settings, user_id: int, status: str, age_days: int) -> uuid.UUID:
    job_id = uuid.uuid4()
    db.create_job(connection, job_id, user_id, "x.pdf", uuid.uuid4().hex, 10, 1)
    finished = datetime.now(UTC) - timedelta(days=age_days)
    connection.execute("UPDATE jobs SET status = %s, finished_at = %s WHERE id = %s", (status, finished, job_id))
    folder = job_dir(settings, job_id)
    (folder / "charts").mkdir(parents=True)
    (folder / "input.pdf").write_bytes(b"%PDF-")
    (folder / "charts" / "chart.png").write_bytes(b"png")
    return job_id


def exists(connection: Any, job_id: uuid.UUID) -> bool:
    return connection.execute("SELECT 1 FROM jobs WHERE id = %s", (job_id,)).fetchone() is not None


def test_nothing_is_removed_by_default(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    user = auth.create_user(connection, "sam", "correct horse battery")
    old = add_job(connection, settings, user, "accepted", age_days=3000)
    assert run_retention(connection, settings) == (0, 0)
    assert exists(connection, old)
    assert (job_dir(settings, old) / "input.pdf").exists()


def test_old_files_are_removed_but_the_result_and_the_row_stay(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_files_days=30)
    user = auth.create_user(connection, "sam", "correct horse battery")
    old = add_job(connection, settings, user, "accepted", age_days=40)
    recent = add_job(connection, settings, user, "accepted", age_days=5)
    assert run_retention(connection, settings) == (1, 0)
    assert not job_dir(settings, old).exists()
    assert exists(connection, old)
    assert connection.execute("SELECT files_deleted_at FROM jobs WHERE id = %s", (old,)).fetchone()["files_deleted_at"]
    assert (job_dir(settings, recent) / "input.pdf").exists()


def test_files_are_only_removed_once(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_files_days=30)
    user = auth.create_user(connection, "sam", "correct horse battery")
    add_job(connection, settings, user, "accepted", age_days=40)
    assert run_retention(connection, settings) == (1, 0)
    assert run_retention(connection, settings) == (0, 0)


def test_old_records_are_removed_with_their_files_and_audit_trail(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_records_days=365)
    user = auth.create_user(connection, "sam", "correct horse battery")
    old = add_job(connection, settings, user, "reviewed", age_days=400)
    keep = add_job(connection, settings, user, "reviewed", age_days=100)
    connection.execute(
        "INSERT INTO review_log (job_id, user_id, field, old_value, new_value) VALUES (%s, %s, 'x', '1', '2')",
        (old, user),
    )
    assert run_retention(connection, settings) == (0, 1)
    assert not exists(connection, old)
    assert not job_dir(settings, old).exists()
    assert connection.execute("SELECT count(*) AS n FROM review_log").fetchone()["n"] == 0
    assert exists(connection, keep)


@pytest.mark.parametrize("status", ["needs_review", "queued", "running"])
def test_a_document_that_is_not_settled_is_never_removed(connection: Any, tmp_path: Path, status: str) -> None:
    settings = settings_for(tmp_path, retention_files_days=1, retention_records_days=1)
    user = auth.create_user(connection, "sam", "correct horse battery")
    job = add_job(connection, settings, user, status, age_days=3000)
    assert run_retention(connection, settings) == (0, 0)
    assert exists(connection, job)
    assert (job_dir(settings, job) / "input.pdf").exists()


def test_a_failed_job_is_removed_like_any_other(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_records_days=30)
    user = auth.create_user(connection, "sam", "correct horse battery")
    job = add_job(connection, settings, user, "failed", age_days=60)
    assert run_retention(connection, settings) == (0, 1)
    assert not exists(connection, job)


def test_remembered_invoices_go_with_their_job(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_records_days=30)
    user = auth.create_user(connection, "sam", "correct horse battery")
    job = add_job(connection, settings, user, "accepted", age_days=60)
    db.record_invoice(
        connection, job, user, "accepted", {}, {"vendor_name": {"value": "Acme"}, "invoice_number": {"value": "A-1"}}
    )
    assert connection.execute("SELECT count(*) AS n FROM invoice_records").fetchone()["n"] == 1
    run_retention(connection, settings)
    assert connection.execute("SELECT count(*) AS n FROM invoice_records").fetchone()["n"] == 0


def test_a_missing_folder_is_not_an_error(connection: Any, tmp_path: Path) -> None:
    settings = settings_for(tmp_path, retention_records_days=30)
    user = auth.create_user(connection, "sam", "correct horse battery")
    job = add_job(connection, settings, user, "accepted", age_days=60)
    import shutil

    shutil.rmtree(job_dir(settings, job))
    assert run_retention(connection, settings) == (0, 1)
