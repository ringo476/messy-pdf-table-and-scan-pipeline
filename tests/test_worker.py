"""A result must never be saved under the wrong job. These tests run real child processes."""

import os
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from pdfpipe.config import Settings
from pdfpipe.service import auth, db
from pdfpipe.service.paths import job_dir
from pdfpipe.service.worker import Extractor, handle_job


def slow_child(pipe: Any) -> None:
    """Stands in for the real extraction process, without loading the models."""
    while True:
        request_id, pdf_path, _, _ = pipe.recv()
        time.sleep(1.0)
        pipe.send((request_id, "done", {"status": "accepted", "result_for": pdf_path}))


def wrong_id_child(pipe: Any) -> None:
    """Behaves as if a reply to an earlier job arrived late."""
    while True:
        pipe.recv()
        pipe.send(("an-earlier-request", "done", {"status": "accepted", "result_for": "another user's file"}))


def crashing_child(pipe: Any) -> None:
    pipe.recv()
    os._exit(1)


class StandInExtractor(Extractor):
    """The real Extractor logic, with a stand-in child process."""

    def __init__(self, settings: Settings, target: Any = slow_child) -> None:
        super().__init__(settings)
        self.target = target
        self.starts = 0
        self.tick_seconds = 0.2

    def start(self) -> None:
        self.stop()
        parent_end, child_end = self.context.Pipe()
        self.process = self.context.Process(target=self.target, args=(child_end,), daemon=True)
        self.process.start()
        child_end.close()
        self.pipe = parent_end
        self.starts += 1


@pytest.fixture
def extractor(settings: Settings) -> Any:
    instance = StandInExtractor(settings)
    yield instance
    instance.stop()


def database_down() -> bool:
    raise psycopg.OperationalError("database connection lost")


def test_a_normal_job_gets_its_own_result(extractor: StandInExtractor) -> None:
    status, value = extractor.run(Path("a.pdf"), Path("a"), lambda: True)
    assert status == "done"
    assert value["result_for"] == "a.pdf"


def test_the_child_is_reused_between_normal_jobs(extractor: StandInExtractor) -> None:
    extractor.run(Path("a.pdf"), Path("a"), lambda: True)
    extractor.run(Path("b.pdf"), Path("b"), lambda: True)
    assert extractor.starts == 1


def test_the_next_job_never_gets_the_aborted_jobs_result(extractor: StandInExtractor) -> None:
    """The regression test for the bug: job B used to receive job A's result here."""
    with pytest.raises(psycopg.OperationalError):
        extractor.run(Path("user_a.pdf"), Path("a"), database_down)
    status, value = extractor.run(Path("user_b.pdf"), Path("b"), lambda: True)
    assert status == "done"
    assert value["result_for"] == "user_b.pdf"


def test_an_aborted_job_kills_the_busy_child(extractor: StandInExtractor) -> None:
    extractor.start()
    busy_child = extractor.process
    with pytest.raises(psycopg.OperationalError):
        extractor.run(Path("a.pdf"), Path("a"), database_down)
    assert extractor.process is None
    assert not busy_child.is_alive()


def test_a_reply_to_another_request_is_rejected(settings: Settings) -> None:
    extractor = StandInExtractor(settings, wrong_id_child)
    try:
        status, value = extractor.run(Path("a.pdf"), Path("a"), lambda: True)
        assert status == "crashed"
        assert "another user" not in str(value)
        assert extractor.process is None
    finally:
        extractor.stop()


def test_a_crashed_child_is_reported(settings: Settings) -> None:
    extractor = StandInExtractor(settings, crashing_child)
    try:
        assert extractor.run(Path("a.pdf"), Path("a"), lambda: True)[0] == "crashed"
    finally:
        extractor.stop()


def test_a_lost_job_stops_the_child(extractor: StandInExtractor) -> None:
    status, _ = extractor.run(Path("a.pdf"), Path("a"), lambda: False)
    assert status == "lost"
    assert extractor.process is None


def test_a_job_over_its_time_limit_stops_the_child(tmp_path: Path) -> None:
    extractor = StandInExtractor(Settings(_env_file=None, job_timeout_seconds=0))
    try:
        assert extractor.run(Path("a.pdf"), Path("a"), lambda: True)[0] == "timeout"
        assert extractor.process is None
    finally:
        extractor.stop()


DATABASE_URL = os.environ.get("PDFPIPE_TEST_DATABASE_URL")


@pytest.mark.postgres
@pytest.mark.skipif(not DATABASE_URL, reason="PDFPIPE_TEST_DATABASE_URL isn't set")
def test_a_dropped_database_can_not_put_one_users_result_on_anothers_job(tmp_path: Path) -> None:
    """The whole path, with two users, a real queue, and a connection that really closes mid-job."""
    settings = Settings(_env_file=None, database_url=DATABASE_URL, data_dir=tmp_path, job_lease_seconds=60)
    extractor = StandInExtractor(settings)
    try:
        with db.connect(settings) as setup:
            db.apply_schema(setup)
            setup.execute("TRUNCATE users, sessions, jobs, review_log CASCADE")
            job_ids = {}
            for name in ("user_a", "user_b"):
                user_id = auth.create_user(setup, name, "correct horse battery")
                job_id = uuid.uuid4()
                db.create_job(setup, job_id, user_id, f"{name}.pdf", uuid.uuid4().hex, 10, 1)
                job_dir(settings, job_id).mkdir(parents=True)
                (job_dir(settings, job_id) / "input.pdf").write_bytes(b"%PDF-")
                job_ids[name] = job_id
                time.sleep(0.01)

            # Job A is running when its worker's database connection dies.
            doomed = db.connect(settings)
            job_a = db.claim_next_job(doomed, "worker", 60)
            assert job_a is not None
            assert job_a["id"] == job_ids["user_a"]
            doomed.close()
            with pytest.raises(psycopg.OperationalError):
                handle_job(job_a, doomed, extractor, settings, "worker")

            # The worker reconnects and takes job B.
            job_b = db.claim_next_job(setup, "worker", 60)
            assert job_b is not None
            assert job_b["id"] == job_ids["user_b"]
            assert handle_job(job_b, setup, extractor, settings, "worker") == "accepted"

            row = setup.execute("SELECT result FROM jobs WHERE id = %s", (job_ids["user_b"],)).fetchone()
            assert row is not None
            assert "user_a" not in str(row["result"])
            assert str(job_ids["user_b"]) in row["result"]["machine"]["result_for"]

            # Job A wasn't given anyone's result. Its lease will expire, and it will run again.
            row = setup.execute("SELECT status, result FROM jobs WHERE id = %s", (job_ids["user_a"],)).fetchone()
            assert row == {"status": "running", "result": None}
    finally:
        extractor.stop()
