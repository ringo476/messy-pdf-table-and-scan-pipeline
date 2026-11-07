"""A background worker. Each one claims one job at a time from the Postgres queue.

The extraction runs in a child process that holds the models. If a PDF hangs or crashes that process,
the parent kills it, reports the failure, and starts a fresh one. A bad file can't take the worker down.
"""

import logging
import multiprocessing as mp
import os
import shutil
import signal
import socket
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import psycopg

from ..config import Settings
from ..engines import Engines
from ..fields.llm import build_resolver
from ..pipeline.intake import RejectedFile
from ..process import process_pdf, result_to_dict
from . import db

logger = logging.getLogger("pdfpipe.worker")

_POLL_SECONDS = 1.0


def job_dir(settings: Settings, job_id: uuid.UUID | str) -> Path:
    return settings.data_dir / "jobs" / str(job_id)


def _child_main(pipe: Any, settings: Settings) -> None:
    """Runs in the child process: loads the models once, then handles one job per request."""
    # Ctrl+C and container shutdown are handled by the parent, which finishes cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    engines = Engines(settings)
    engines.warm_up()
    resolver = build_resolver(settings)
    pipe.send(("ready", None))
    while True:
        request = pipe.recv()
        if request is None:
            return
        request_id, pdf_path, output_dir, llm_budget_usd = request
        try:
            result = process_pdf(Path(pdf_path), settings, engines, Path(output_dir), resolver, llm_budget_usd)
            pipe.send((request_id, "done", result_to_dict(result)))
        except RejectedFile as error:
            pipe.send((request_id, "rejected", str(error)))
        except Exception:  # Any failure is reported to the parent, which decides whether to retry.
            logger.exception("processing failed for %s", pdf_path)
            pipe.send((request_id, "error", traceback.format_exc(limit=3)))


class Extractor:
    """Owns the child process, and replaces it whenever it has to be killed."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.context = mp.get_context("spawn")
        self.process: Any = None
        self.pipe: Any = None
        self.tick_seconds = 5.0

    def start(self) -> None:
        self.stop()
        parent_end, child_end = self.context.Pipe()
        self.process = self.context.Process(target=_child_main, args=(child_end, self.settings), daemon=True)
        self.process.start()
        child_end.close()
        self.pipe = parent_end
        # The first start can include loading large model files.
        if not parent_end.poll(600):
            raise RuntimeError("the models did not load in time")
        status, _ = parent_end.recv()
        if status != "ready":
            raise RuntimeError("the models failed to load")

    def stop(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.kill()
            self.process.join(10)
        self.process, self.pipe = None, None

    def run(
        self, pdf_path: Path, output_dir: Path, on_tick: Any, llm_budget_usd: float | None = None
    ) -> tuple[str, Any]:
        """Runs one job. on_tick is called regularly, and returning False from it cancels the job.
        llm_budget_usd is the most Claude may cost on this document, or None for no extra limit.

        A busy child must never be reused. If this method exits abnormally while the child is still working,
        the child's late reply would otherwise be read as the next job's result, and could show one user's
        document to another. So any abnormal exit kills the child, and each reply must carry its request's ID.
        """
        if self.process is None or not self.process.is_alive() or self.pipe is None:
            self.start()
        assert self.pipe is not None
        crashed = "The extraction process stopped unexpectedly on this file."
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.settings.job_timeout_seconds
        try:
            self.pipe.send((request_id, str(pdf_path), str(output_dir), llm_budget_usd))
            while True:
                if self.pipe.poll(self.tick_seconds):
                    reply_id, status, value = self.pipe.recv()
                    if reply_id != request_id:
                        logger.error("the extraction process replied to the wrong request, so it was replaced")
                        self.stop()
                        return "crashed", crashed
                    return str(status), value
                if not self.process.is_alive():
                    self.stop()
                    return "crashed", crashed
                if time.monotonic() > deadline:
                    self.stop()
                    return "timeout", f"Processing took longer than {self.settings.job_timeout_seconds} seconds."
                if not on_tick():
                    self.stop()
                    return "lost", "This worker lost the job."
        except (EOFError, OSError):
            self.stop()
            return "crashed", crashed
        except BaseException:
            # For example, the database dropped during a lease renewal. The job's lease then expires,
            # and a worker picks it up again.
            self.stop()
            raise


def handle_job(
    job: dict[str, Any], connection: db.Connection, extractor: Extractor, settings: Settings, worker: str
) -> str:
    job_id = job["id"]
    folder = job_dir(settings, job_id)
    pdf_path = folder / "input.pdf"

    def still_ours() -> bool:
        return db.extend_lease(connection, job_id, worker, settings.job_lease_seconds)

    if not pdf_path.exists():
        db.fail_or_retry_job(
            connection,
            job_id,
            worker,
            "The uploaded file is missing from disk.",
            settings.job_max_attempts,
            retry=False,
        )
        return "failed"

    budget = None
    if settings.llm_enabled:
        # What is left of today's Claude budget, shared by all workers through the database.
        budget = max(0.0, settings.llm_daily_budget_usd - db.llm_spend_today(connection))
    status, value = extractor.run(pdf_path, folder, still_ours, budget)
    if status == "done":
        db.complete_job(connection, job_id, worker, value)
        return str(value["status"])
    if status == "rejected":
        db.fail_or_retry_job(connection, job_id, worker, str(value), settings.job_max_attempts, retry=False)
        return "failed"
    if status == "lost":
        return "lost"
    # A timeout would just time out again, so it fails right away. Crashes and errors may be temporary.
    retry = status != "timeout"
    logger.warning("job %s: %s", job_id, status)
    message = str(value) if status in ("timeout", "crashed") else "Something went wrong while processing this file."
    db.fail_or_retry_job(connection, job_id, worker, message, settings.job_max_attempts, retry=retry)
    return "failed" if not retry else "retrying"


def copy_reused_files(settings: Settings, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
    source = job_dir(settings, source_id) / "charts"
    if source.is_dir():
        shutil.copytree(source, job_dir(settings, job_id) / "charts", dirs_exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("rapid_layout").setLevel(logging.WARNING)
    logging.getLogger("RapidTable").setLevel(logging.WARNING)
    settings = Settings()
    worker = f"{socket.gethostname()}-{os.getpid()}"
    stopping = {"now": False}

    def request_stop(*_: Any) -> None:
        logger.info("stopping after the current job")
        stopping["now"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    connection = _connect_with_retry(settings)
    db.apply_schema(connection)
    extractor = Extractor(settings)
    extractor.start()
    logger.info("worker %s is ready", worker)
    last_recovery = 0.0

    try:
        while not stopping["now"]:
            try:
                if time.monotonic() - last_recovery > 15:
                    recovered = db.requeue_expired_jobs(connection, settings.job_max_attempts)
                    if recovered:
                        logger.warning("recovered %d jobs from stopped workers", recovered)
                    last_recovery = time.monotonic()
                job = db.claim_next_job(connection, worker, settings.job_lease_seconds)
                if job is None:
                    time.sleep(_POLL_SECONDS)
                    continue
                started = time.perf_counter()
                outcome = handle_job(job, connection, extractor, settings, worker)
                logger.info(
                    "job %s pages=%s outcome=%s seconds=%.1f",
                    job["id"],
                    job["page_count"],
                    outcome,
                    time.perf_counter() - started,
                )
            except psycopg.OperationalError:
                # The database restarted. Reconnect, and the lease system recovers any lost job.
                logger.warning("lost the database connection, reconnecting")
                connection = _connect_with_retry(settings)
    finally:
        extractor.stop()
        connection.close()


def _connect_with_retry(settings: Settings, attempts: int = 60) -> db.Connection:
    for attempt in range(attempts):
        try:
            return db.connect(settings)
        except psycopg.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()
