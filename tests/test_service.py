"""Service tests against a real Postgres. Set PDFPIPE_TEST_DATABASE_URL to run them."""

import io
import os
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from pdfpipe import __version__
from pdfpipe.config import Settings
from pdfpipe.service import auth, db, worker
from pdfpipe.service.app import create_app

DATABASE_URL = os.environ.get("PDFPIPE_TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DATABASE_URL, reason="PDFPIPE_TEST_DATABASE_URL isn't set")]
PASSWORD = "correct horse battery"


def machine_result(status: str = "needs_review") -> dict[str, Any]:
    return {
        "status": status,
        "pipeline_version": __version__,
        "fingerprint": "x",
        "page_count": 1,
        "pages": [],
        "tables": [],
        "fields": {
            "invoice_number": {"value": "INV-1", "needs_review": False, "confidence": 0.97, "review_reasons": []},
            "grand_total": {
                "value": "10.00",
                "needs_review": True,
                "confidence": 0.5,
                "review_reasons": ["sum is wrong"],
            },
        },
    }


def pdf_bytes(text: str = "hello") -> bytes:
    buffer = io.BytesIO()
    # invariant=1 leaves out the timestamp, so the same text always gives the same file.
    page = canvas.Canvas(buffer, invariant=1)
    page.drawString(100, 100, text)
    page.save()
    return buffer.getvalue()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, database_url=DATABASE_URL, data_dir=tmp_path, max_active_jobs_per_user=3, max_upload_mb=1
    )


@pytest.fixture
def connection(test_settings: Settings) -> Iterator[db.Connection]:
    with db.connect(test_settings) as conn:
        db.apply_schema(conn)
        conn.execute("TRUNCATE users, sessions, jobs, review_log CASCADE")
        yield conn


@pytest.fixture
def app_client(test_settings: Settings, connection: db.Connection) -> Iterator[TestClient]:
    with TestClient(create_app(test_settings)) as client:
        yield client


def logged_in(app: Any, connection: db.Connection, name: str) -> TestClient:
    auth.create_user(connection, name, PASSWORD)
    client = TestClient(app)
    assert client.post("/api/login", json={"username": name, "password": PASSWORD}).status_code == 200
    return client


def upload(client: TestClient, text: str = "hello", name: str = "a.pdf") -> Any:
    return client.post("/api/jobs", files={"file": (name, pdf_bytes(text), "application/pdf")})


def add_job(connection: db.Connection, user_id: int, status: str = "queued") -> uuid.UUID:
    job_id = uuid.uuid4()
    db.create_job(connection, job_id, user_id, "x.pdf", uuid.uuid4().hex, 10, 1)
    if status != "queued":
        connection.execute("UPDATE jobs SET status = %s WHERE id = %s", (status, job_id))
    return job_id


class TestLogin:
    def test_correct_and_wrong_password(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        assert auth.check_login(connection, "sam", PASSWORD) == user_id
        assert auth.check_login(connection, "sam", "wrong") is None
        assert auth.check_login(connection, "nobody", PASSWORD) is None

    def test_password_is_never_stored_as_plain_text(self, connection: db.Connection) -> None:
        auth.create_user(connection, "sam", PASSWORD)
        row = connection.execute("SELECT password_hash FROM users").fetchone()
        assert row is not None
        assert PASSWORD not in row["password_hash"]

    def test_session_token_is_stored_only_as_a_hash(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        token = auth.start_session(connection, user_id, 1)
        row = connection.execute("SELECT token_hash FROM sessions").fetchone()
        assert row is not None
        assert row["token_hash"] != token
        assert auth.user_for_session(connection, token)["id"] == user_id  # type: ignore[index]

    def test_expired_session_does_not_work(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        token = auth.start_session(connection, user_id, 1)
        connection.execute("UPDATE sessions SET expires_at = now() - interval '1 minute'")
        assert auth.user_for_session(connection, token) is None

    def test_no_session_means_no_access(self, app_client: TestClient) -> None:
        assert app_client.get("/api/jobs").status_code == 401
        assert app_client.post("/api/jobs", files={"file": ("a.pdf", pdf_bytes())}).status_code == 401

    def test_logout_ends_the_session(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        assert client.get("/api/me").status_code == 200
        client.post("/api/logout")
        assert client.get("/api/me").status_code == 401

    def test_repeated_failures_are_blocked(self, app_client: TestClient, connection: db.Connection) -> None:
        auth.create_user(connection, "sam", PASSWORD)
        for _ in range(8):
            assert app_client.post("/api/login", json={"username": "sam", "password": "nope"}).status_code == 401
        assert app_client.post("/api/login", json={"username": "sam", "password": PASSWORD}).status_code == 429


class TestIsolation:
    def test_one_user_can_not_touch_another_users_job(self, app_client: TestClient, connection: db.Connection) -> None:
        owner = logged_in(app_client.app, connection, "owner")
        other = logged_in(app_client.app, connection, "other")
        job_id = upload(owner).json()["id"]
        assert owner.get(f"/api/jobs/{job_id}").status_code == 200
        assert other.get(f"/api/jobs/{job_id}").status_code == 404
        assert other.get(f"/api/jobs/{job_id}/result.json").status_code == 404
        assert other.get(f"/api/jobs/{job_id}/charts/x.png").status_code == 404
        assert other.post(f"/api/jobs/{job_id}/review", json={"approve": []}).status_code == 404
        assert other.delete(f"/api/jobs/{job_id}").status_code == 404
        assert other.get("/api/jobs").json() == []
        assert owner.get(f"/api/jobs/{job_id}").status_code == 200

    def finished_and_corrected(self, client: TestClient, connection: db.Connection) -> None:
        first_id = upload(client, "same").json()["id"]
        connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (first_id,))
        assert db.complete_job(connection, uuid.UUID(first_id), "w", machine_result())
        edit = {"edits": [{"name": "invoice_number", "value": "SECRET-1"}], "approve": ["grand_total"], "finish": True}
        assert client.post(f"/api/jobs/{first_id}/review", json=edit).status_code == 200

    def test_reuse_never_copies_corrections(self, app_client: TestClient, connection: db.Connection) -> None:
        user = logged_in(app_client.app, connection, "first")
        self.finished_and_corrected(user, connection)

        reused = upload(user, "same").json()
        assert reused["reused"] is True
        job = user.get(f"/api/jobs/{reused['id']}").json()
        assert "reviewed" not in job["result"]
        assert "SECRET" not in str(job)
        assert job["status"] == "needs_review"

    def test_another_user_is_never_told_a_file_was_seen_before(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        first = logged_in(app_client.app, connection, "first")
        second = logged_in(app_client.app, connection, "second")
        self.finished_and_corrected(first, connection)

        reply = upload(second, "same").json()
        assert reply["reused"] is False
        assert reply["status"] == "queued"


class TestUpload:
    def test_accepts_a_pdf_and_queues_it(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        reply = upload(client)
        assert reply.status_code == 202
        assert reply.json()["status"] == "queued"
        assert client.get("/api/jobs").json()[0]["status"] == "queued"

    def test_rejects_a_file_that_is_not_a_pdf(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        reply = client.post("/api/jobs", files={"file": ("evil.pdf", b"MZ not a pdf", "application/pdf")})
        assert reply.status_code == 422
        assert client.get("/api/jobs").json() == []

    def test_rejects_an_oversized_file_and_leaves_nothing_behind(
        self, app_client: TestClient, connection: db.Connection, test_settings: Settings
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        reply = client.post("/api/jobs", files={"file": ("big.pdf", b"%PDF-" + b"x" * (2 * 1024 * 1024))})
        assert reply.status_code == 413
        assert list((test_settings.data_dir / "uploads").iterdir()) == []

    def test_a_path_in_the_filename_is_harmless(
        self, app_client: TestClient, connection: db.Connection, test_settings: Settings
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = upload(client, name="../../../evil.pdf").json()["id"]
        assert client.get(f"/api/jobs/{job_id}").json()["filename"] == "evil.pdf"
        assert (test_settings.data_dir / "jobs" / job_id / "input.pdf").exists()

    def test_limit_on_waiting_documents(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        for index in range(3):
            assert upload(client, f"doc {index}").status_code == 202
        assert upload(client, "one more").status_code == 429

    def test_chart_route_refuses_path_tricks(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = upload(client).json()["id"]
        assert client.get(f"/api/jobs/{job_id}/charts/..%2Finput.pdf").status_code == 404

    def test_delete_removes_the_row_and_the_files(
        self, app_client: TestClient, connection: db.Connection, test_settings: Settings
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = upload(client).json()["id"]
        assert client.delete(f"/api/jobs/{job_id}").status_code == 204
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert not (test_settings.data_dir / "jobs" / job_id).exists()


class TestReview:
    def finished_job(self, client: TestClient, connection: db.Connection) -> str:
        job_id = upload(client).json()["id"]
        connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))
        db.complete_job(connection, uuid.UUID(job_id), "w", machine_result())
        return str(job_id)

    def test_can_not_finish_with_open_fields(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.finished_job(client, connection)
        assert client.post(f"/api/jobs/{job_id}/review", json={"finish": True}).status_code == 409

    def test_correction_is_logged_and_the_original_is_kept(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.finished_job(client, connection)
        reply = client.post(
            f"/api/jobs/{job_id}/review", json={"edits": [{"name": "grand_total", "value": "11.00"}], "finish": True}
        )
        assert reply.status_code == 200
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "reviewed"
        assert job["result"]["machine"]["fields"]["grand_total"]["value"] == "10.00"
        assert job["result"]["reviewed"]["fields"]["grand_total"]["value"] == "11.00"
        assert job["history"][0]["old_value"] == "10.00"
        assert job["history"][0]["new_value"] == "11.00"
        assert job["history"][0]["username"] == "sam"
        assert client.get(f"/api/jobs/{job_id}/result.json").json()["fields"]["grand_total"]["value"] == "11.00"

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("grand_total", "abc"),
            ("grand_total", "-5.00"),
            ("grand_total", "1.234"),
            ("invoice_date", "03/04/2026"),
            ("invoice_date", "1999-01-01"),
            ("invoice_number", "bad id!"),
            ("invoice_number", ""),
            ("line_items", "anything"),
        ],
    )
    def test_a_typed_value_must_pass_the_same_format_checks(
        self, app_client: TestClient, connection: db.Connection, name: str, value: str
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.finished_job(client, connection)
        reply = client.post(f"/api/jobs/{job_id}/review", json={"edits": [{"name": name, "value": value}]})
        assert reply.status_code == 422

    def test_a_rejected_edit_changes_nothing(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.finished_job(client, connection)
        edits = [{"name": "invoice_number", "value": "INV-2"}, {"name": "grand_total", "value": "abc"}]
        assert client.post(f"/api/jobs/{job_id}/review", json={"edits": edits}).status_code == 422
        assert client.get(f"/api/jobs/{job_id}").json()["history"] == []

    def test_edited_amounts_must_still_add_up(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = upload(client).json()["id"]
        machine = machine_result()
        machine["fields"]["subtotal"] = {
            "value": "8.00",
            "needs_review": False,
            "confidence": 0.97,
            "review_reasons": [],
        }
        machine["fields"]["tax"] = {"value": "2.00", "needs_review": False, "confidence": 0.97, "review_reasons": []}
        connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))
        db.complete_job(connection, uuid.UUID(job_id), "w", machine)

        wrong = client.post(f"/api/jobs/{job_id}/review", json={"edits": [{"name": "grand_total", "value": "99.00"}]})
        assert wrong.status_code == 422
        assert "subtotal plus tax" in wrong.json()["detail"]
        right = client.post(f"/api/jobs/{job_id}/review", json={"edits": [{"name": "grand_total", "value": "10.00"}]})
        assert right.status_code == 200

    def test_a_date_is_stored_in_one_format(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = upload(client).json()["id"]
        machine = machine_result()
        machine["fields"]["invoice_date"] = {
            "value": None,
            "needs_review": True,
            "confidence": 0.0,
            "review_reasons": [],
        }
        connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))
        db.complete_job(connection, uuid.UUID(job_id), "w", machine)
        edits = [{"name": "invoice_date", "value": "5 March 2026"}]
        assert client.post(f"/api/jobs/{job_id}/review", json={"edits": edits}).status_code == 200
        fields = client.get(f"/api/jobs/{job_id}").json()["result"]["reviewed"]["fields"]
        assert fields["invoice_date"]["value"] == "2026-03-05"

    def job_with_items(self, client: TestClient, connection: db.Connection) -> str:
        job_id = upload(client).json()["id"]
        machine = machine_result()
        machine["fields"]["subtotal"] = {
            "value": "100.00",
            "needs_review": False,
            "confidence": 0.97,
            "review_reasons": [],
        }
        machine["fields"]["grand_total"] = {
            "value": "100.00",
            "needs_review": False,
            "confidence": 0.97,
            "review_reasons": [],
        }
        machine["fields"]["line_items"] = {
            "value": [{"description": "Pens", "quantity": "2", "unit_price": "20.00", "amount": "40.00"}],
            "needs_review": True,
            "confidence": 0.5,
            "review_reasons": ["the line items add up to 40.00, but the subtotal is 100.00"],
        }
        connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))
        db.complete_job(connection, uuid.UUID(job_id), "w", machine)
        return str(job_id)

    def test_line_items_can_be_edited_and_the_change_is_logged(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.job_with_items(client, connection)
        lines = [
            {"description": "Pens", "quantity": "2", "unit_price": "20.00", "amount": "40.00"},
            {"description": "Paper", "quantity": "3", "unit_price": "20.00", "amount": "60.00"},
        ]
        reply = client.post(
            f"/api/jobs/{job_id}/review",
            json={"edits": [{"name": "line_items", "value": lines}], "approve": ["grand_total"]},
        )
        assert reply.status_code == 200
        job = client.get(f"/api/jobs/{job_id}").json()
        reviewed = job["result"]["reviewed"]["fields"]["line_items"]
        assert [item["description"] for item in reviewed["value"]] == ["Pens", "Paper"]
        assert reviewed["source"] == "person"
        assert job["result"]["machine"]["fields"]["line_items"]["value"][0]["description"] == "Pens"
        assert len(job["result"]["machine"]["fields"]["line_items"]["value"]) == 1  # the original is kept
        history = [entry for entry in job["history"] if entry["field"] == "line_items"]
        assert len(history) == 1
        assert len(history[0]["old_value"]) == 1
        assert len(history[0]["new_value"]) == 2
        assert (
            client.get(f"/api/jobs/{job_id}/result.json").json()["fields"]["line_items"]["value"][1]["amount"]
            == "60.00"
        )

    def test_line_items_that_do_not_add_up_are_refused_and_nothing_is_saved(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.job_with_items(client, connection)
        lines = [{"description": "Pens", "quantity": "2", "unit_price": "20.00", "amount": "40.00"}]
        reply = client.post(f"/api/jobs/{job_id}/review", json={"edits": [{"name": "line_items", "value": lines}]})
        assert reply.status_code == 422
        assert "subtotal" in reply.json()["detail"]
        assert client.get(f"/api/jobs/{job_id}").json()["history"] == []

    def test_a_user_cannot_edit_another_users_line_items(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        owner = logged_in(app_client.app, connection, "owner")
        other = logged_in(app_client.app, connection, "other")
        job_id = self.job_with_items(owner, connection)
        reply = other.post(f"/api/jobs/{job_id}/review", json={"edits": [{"name": "line_items", "value": []}]})
        assert reply.status_code == 404

    def test_unknown_field_is_refused(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = self.finished_job(client, connection)
        assert client.post(f"/api/jobs/{job_id}/review", json={"approve": ["nope"]}).status_code == 422


def stored_invoice(
    connection: db.Connection,
    client: TestClient,
    text: str,
    number: str,
    *,
    vendor: str = "Acme Office Supply",
    total: str = "110.00",
    date: str = "2026-03-05",
    status: str = "accepted",
    currency: str = "USD",
    document_type: str = "invoice",
) -> uuid.UUID:
    """Uploads a file, finishes its job with the given values, and remembers it, as a worker does."""
    job_id = uuid.UUID(upload(client, text).json()["id"])
    connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))

    def field(value: Any) -> dict[str, Any]:
        return {"value": value, "needs_review": False, "confidence": 0.97, "review_reasons": [], "checks": []}

    machine = {
        **machine_result(status),
        "document_type": document_type,
        "fields": {
            "vendor_name": field(vendor),
            "invoice_number": field(number),
            "invoice_date": field(date),
            "currency": field(currency),
            "grand_total": field(total),
        },
    }
    machine = worker.check_against_history(
        connection, db.get_job(connection, job_id, owner_of(connection, job_id)), machine
    )  # type: ignore[arg-type]
    assert db.complete_job(connection, job_id, "w", machine)
    job = db.get_job(connection, job_id, owner_of(connection, job_id))
    assert job is not None
    worker.remember(connection, job, machine)
    return job_id


def owner_of(connection: db.Connection, job_id: uuid.UUID) -> int:
    row = connection.execute("SELECT user_id FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    return int(row["user_id"])


def status_of(connection: db.Connection, job_id: uuid.UUID) -> str:
    row = connection.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    return str(row["status"])


class TestInvoiceMemory:
    def test_a_second_copy_of_an_invoice_is_flagged(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        stored_invoice(connection, client, "first", "INV-1001")
        second = stored_invoice(connection, client, "second", "INV-1001")
        assert status_of(connection, second) == "needs_review"
        result = client.get(f"/api/jobs/{second}").json()["result"]["machine"]
        assert result["history_checked"] is True
        assert result["fields"]["invoice_number"]["needs_review"] is True

    def test_a_different_number_is_not_flagged(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        stored_invoice(connection, client, "first", "INV-1001")
        assert status_of(connection, stored_invoice(connection, client, "second", "INV-1002")) == "accepted"

    def test_another_users_invoices_are_never_used(self, app_client: TestClient, connection: db.Connection) -> None:
        first = logged_in(app_client.app, connection, "first")
        second = logged_in(app_client.app, connection, "second")
        stored_invoice(connection, first, "same-a", "INV-1001")
        job = stored_invoice(connection, second, "same-b", "INV-1001")
        assert status_of(connection, job) == "accepted"

    def test_a_number_that_breaks_the_vendors_pattern_is_flagged(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        for index in range(4):
            stored_invoice(connection, client, f"doc{index}", f"INV-100{index}")
        odd = stored_invoice(connection, client, "odd", "INV-l005")
        assert status_of(connection, odd) == "needs_review"

    def test_a_vendor_name_with_a_small_ocr_slip_still_matches(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        stored_invoice(connection, client, "first", "INV-1001")
        slip = stored_invoice(connection, client, "second", "INV-1001", vendor="Acme Offlce Supply Ltd")
        assert status_of(connection, slip) == "needs_review"

    def test_deleting_a_job_removes_what_was_remembered(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = stored_invoice(connection, client, "first", "INV-1001")
        assert client.delete(f"/api/jobs/{job_id}").status_code == 204
        row = connection.execute("SELECT count(*) AS n FROM invoice_records").fetchone()
        assert row is not None
        assert row["n"] == 0

    def test_finishing_a_review_remembers_the_corrected_values(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        job_id = stored_invoice(connection, client, "first", "INV-1001", status="needs_review")
        assert client.get("/api/invoices").json()["invoices"] == []  # not trusted until a person has confirmed it
        reply = client.post(
            f"/api/jobs/{job_id}/review",
            json={"edits": [{"name": "grand_total", "value": "125.00"}], "finish": True},
        )
        assert reply.status_code == 200
        found = client.get("/api/invoices").json()["invoices"]
        assert [(row["invoice_number"], row["grand_total"]) for row in found] == [("INV-1001", "125.00")]


class TestInvoiceSearch:
    def fill(self, connection: db.Connection, client: TestClient) -> None:
        stored_invoice(connection, client, "a", "A-1", total="100.10", date="2026-07-10")
        stored_invoice(connection, client, "b", "A-2", total="50.05", date="2026-08-20")
        stored_invoice(connection, client, "c", "A-3", total="900.00", date="2026-01-02")
        stored_invoice(connection, client, "d", "O-1", total="20.00", date="2026-08-01", vendor="Orbit Systems Inc")
        stored_invoice(connection, client, "e", "A-9", total="10.00", date="2026-08-21", document_type="credit_note")
        stored_invoice(connection, client, "f", "A-8", total="777.00", date="2026-08-22", status="needs_review")

    def test_a_vendor_and_date_range_are_added_up_by_code(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        self.fill(connection, client)
        reply = client.get(
            "/api/invoices", params={"vendor": "acme", "date_from": "2026-07-01", "date_to": "2026-09-30"}
        )
        body = reply.json()
        assert [row["invoice_number"] for row in body["invoices"]] == ["A-1", "A-2", "A-9"]
        assert body["totals"] == {"USD": "140.15"}  # 100.10 + 50.05 - 10.00 credit note
        assert "3 invoices" in body["summary"]

    def test_an_invoice_still_waiting_for_review_is_not_counted(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        self.fill(connection, client)
        numbers = [row["invoice_number"] for row in client.get("/api/invoices").json()["invoices"]]
        assert "A-8" not in numbers

    def test_every_vendor_when_none_is_named(self, app_client: TestClient, connection: db.Connection) -> None:
        client = logged_in(app_client.app, connection, "sam")
        self.fill(connection, client)
        assert len(client.get("/api/invoices").json()["invoices"]) == 5

    def test_nobody_can_search_another_users_invoices(self, app_client: TestClient, connection: db.Connection) -> None:
        owner = logged_in(app_client.app, connection, "owner")
        other = logged_in(app_client.app, connection, "other")
        self.fill(connection, owner)
        body = other.get("/api/invoices", params={"vendor": "acme"}).json()
        assert body["invoices"] == []
        assert body["totals"] == {}

    def test_search_needs_a_login(self, app_client: TestClient) -> None:
        assert app_client.get("/api/invoices").status_code == 401
        assert app_client.post("/api/ask", json={"question": "x"}).status_code == 401


class TestAsking:
    def asking(self, test_settings: Settings, reply: Any) -> Any:
        from test_history import FakeClaude

        return TestClient(create_app(test_settings, claude_client=FakeClaude(reply)))

    def test_a_question_is_answered_from_the_users_own_invoices(
        self, test_settings: Settings, connection: db.Connection
    ) -> None:
        reply = {"vendor": "Acme", "date_from": "2026-07-01", "date_to": "2026-09-30"}
        with self.asking(test_settings, reply) as base:
            client = logged_in(base.app, connection, "sam")
            TestInvoiceSearch().fill(connection, client)
            body = client.post("/api/ask", json={"question": "What did Acme invoice us last quarter?"}).json()
        assert body["totals"] == {"USD": "140.15"}
        assert body["filters"]["vendor"] == "Acme"
        assert "totalling 140.15 USD" in body["summary"]

    def test_a_question_never_reaches_another_users_invoices(
        self, test_settings: Settings, connection: db.Connection
    ) -> None:
        reply = {"vendor": "Acme", "date_from": None, "date_to": None}
        with self.asking(test_settings, reply) as base:
            owner = logged_in(base.app, connection, "owner")
            other = logged_in(base.app, connection, "other")
            TestInvoiceSearch().fill(connection, owner)
            assert other.post("/api/ask", json={"question": "Acme?"}).json()["invoices"] == []

    def test_a_question_that_cannot_be_understood_is_refused(
        self, test_settings: Settings, connection: db.Connection
    ) -> None:
        with self.asking(test_settings, ["nonsense"]) as base:
            client = logged_in(base.app, connection, "sam")
            assert client.post("/api/ask", json={"question": "hmm"}).status_code == 422

    def test_without_claude_the_filters_still_work_and_asking_says_why(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        assert client.get("/api/me").json()["can_ask"] is False
        assert client.post("/api/ask", json={"question": "anything"}).status_code == 503


class TestClaudeSpend:
    def test_todays_spend_adds_up_and_ignores_reused_results(
        self, app_client: TestClient, connection: db.Connection
    ) -> None:
        client = logged_in(app_client.app, connection, "sam")
        for text, cost in (("one", 0.4), ("two", 0.1)):
            job_id = upload(client, text).json()["id"]
            connection.execute("UPDATE jobs SET status = 'running', worker = 'w' WHERE id = %s", (job_id,))
            db.complete_job(
                connection, uuid.UUID(job_id), "w", {**machine_result(), "llm": {"estimated_cost_usd": cost}}
            )
        assert db.llm_spend_today(connection) == pytest.approx(0.5)

        copy = upload(client, "one").json()
        assert copy["reused"] is True
        assert db.llm_spend_today(connection) == pytest.approx(0.5)


class TestQueue:
    def test_each_job_goes_to_exactly_one_worker(self, connection: db.Connection, test_settings: Settings) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        jobs = {add_job(connection, user_id) for _ in range(40)}
        claimed: list[uuid.UUID] = []
        lock = threading.Lock()

        def worker(name: str) -> None:
            with db.connect(test_settings) as conn:
                while (job := db.claim_next_job(conn, name, 60)) is not None:
                    with lock:
                        claimed.append(job["id"])

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(claimed) == 40
        assert set(claimed) == jobs

    def test_one_users_batch_does_not_block_another_user(self, connection: db.Connection) -> None:
        busy = auth.create_user(connection, "busy", PASSWORD)
        other = auth.create_user(connection, "other", PASSWORD)
        for _ in range(5):
            add_job(connection, busy)
        first = db.claim_next_job(connection, "w1", 60)
        assert first is not None
        assert first["user_id"] == busy
        late = add_job(connection, other)
        second = db.claim_next_job(connection, "w2", 60)
        assert second is not None
        assert second["id"] == late

    def test_a_dead_workers_job_runs_again(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        job_id = add_job(connection, user_id)
        db.claim_next_job(connection, "dead", 60)
        assert db.requeue_expired_jobs(connection, 3) == 0
        connection.execute("UPDATE jobs SET lease_until = now() - interval '1 second'")
        assert db.requeue_expired_jobs(connection, 3) == 1
        job = db.claim_next_job(connection, "alive", 60)
        assert job is not None
        assert job["id"] == job_id
        assert job["attempts"] == 2

    def test_a_file_that_keeps_killing_workers_eventually_fails(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        job_id = add_job(connection, user_id)
        for _ in range(3):
            assert db.claim_next_job(connection, "w", 60) is not None
            connection.execute("UPDATE jobs SET lease_until = now() - interval '1 second'")
            db.requeue_expired_jobs(connection, 3)
        row = connection.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
        assert row is not None
        assert row["status"] == "failed"

    def test_a_worker_that_lost_its_job_can_not_overwrite_the_result(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        job_id = add_job(connection, user_id)
        db.claim_next_job(connection, "slow", 60)
        connection.execute("UPDATE jobs SET lease_until = now() - interval '1 second'")
        db.requeue_expired_jobs(connection, 3)
        db.claim_next_job(connection, "fast", 60)
        assert db.complete_job(connection, job_id, "slow", machine_result()) is False
        assert db.complete_job(connection, job_id, "fast", machine_result()) is True
        assert db.extend_lease(connection, job_id, "slow", 60) is False

    def test_temporary_failure_retries_after_a_delay(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        job_id = add_job(connection, user_id)
        db.claim_next_job(connection, "w", 60)
        db.fail_or_retry_job(connection, job_id, "w", "boom", 3, retry=True)
        row = connection.execute(
            "SELECT status, run_after > now() AS delayed FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        assert row is not None
        assert row["status"] == "queued"
        assert row["delayed"] is True
        assert db.claim_next_job(connection, "w", 60) is None

    def test_permanent_failure_does_not_retry(self, connection: db.Connection) -> None:
        user_id = auth.create_user(connection, "sam", PASSWORD)
        job_id = add_job(connection, user_id)
        db.claim_next_job(connection, "w", 60)
        db.fail_or_retry_job(connection, job_id, "w", "bad file", 3, retry=False)
        row = connection.execute("SELECT status, error FROM jobs WHERE id = %s", (job_id,)).fetchone()
        assert row == {"status": "failed", "error": "bad file"}


def test_duplicate_username_is_refused(connection: db.Connection) -> None:
    auth.create_user(connection, "sam", PASSWORD)
    with pytest.raises(psycopg.errors.UniqueViolation):
        auth.create_user(connection, "sam", PASSWORD)
