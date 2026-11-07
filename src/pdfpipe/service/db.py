"""Postgres access, including the job queue. Postgres is both the database and the queue."""

import json
import uuid
from datetime import datetime
from importlib import resources
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..config import Settings

Connection = psycopg.Connection[dict[str, Any]]

_FINISHED = ("accepted", "needs_review", "reviewed")


def connect(settings: Settings) -> Connection:
    return psycopg.connect(settings.database_url, row_factory=dict_row, autocommit=True)


def apply_schema(connection: Connection) -> None:
    sql = resources.files("pdfpipe.service").joinpath("schema.sql").read_text(encoding="utf-8")
    # A lock stops the web app and several workers from creating the same tables at the same moment.
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(727201)")
        connection.execute(sql)


def active_job_count(connection: Connection, user_id: int) -> int:
    row = connection.execute(
        "SELECT count(*) AS n FROM jobs WHERE user_id = %s AND status IN ('queued', 'running')", (user_id,)
    ).fetchone()
    return int(row["n"]) if row else 0


def create_job(
    connection: Connection,
    job_id: uuid.UUID,
    user_id: int,
    filename: str,
    fingerprint: str,
    size_bytes: int,
    page_count: int,
) -> None:
    connection.execute(
        """INSERT INTO jobs (id, user_id, filename, fingerprint, size_bytes, page_count)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (job_id, user_id, filename, fingerprint, size_bytes, page_count),
    )


def find_reusable(
    connection: Connection, user_id: int, fingerprint: str, pipeline_version: str
) -> dict[str, Any] | None:
    """Finds this user's own earlier automatic result for the same file from the same pipeline version.

    It only looks at the user's own jobs. Looking at everyone's would tell a user whether somebody else
    had already uploaded a particular file.
    The result stored on a job is the machine's output. People's corrections are kept in the job's
    own reviewed_result key, which is never copied.
    """
    return connection.execute(
        """SELECT id, status, result FROM jobs
           WHERE user_id = %s AND fingerprint = %s AND status IN ('accepted', 'needs_review', 'reviewed')
             AND result->'machine'->>'pipeline_version' = %s
           ORDER BY finished_at DESC LIMIT 1""",
        (user_id, fingerprint, pipeline_version),
    ).fetchone()


def llm_spend_today(connection: Connection) -> float:
    """What Claude has cost today, in US dollars. Results copied from an earlier job cost nothing new."""
    row = connection.execute(
        """SELECT coalesce(sum((result->'machine'->'llm'->>'estimated_cost_usd')::float8), 0) AS spent
           FROM jobs
           WHERE finished_at >= date_trunc('day', now()) AND reused_from IS NULL AND result IS NOT NULL"""
    ).fetchone()
    return float(row["spent"]) if row else 0.0


def finish_with_reused_result(
    connection: Connection, job_id: uuid.UUID, source_id: uuid.UUID, machine: dict[str, Any]
) -> None:
    connection.execute(
        """UPDATE jobs SET status = %s, result = %s, reused_from = %s, finished_at = now()
           WHERE id = %s""",
        (machine["status"], Jsonb({"machine": machine}), source_id, job_id),
    )


def claim_next_job(connection: Connection, worker: str, lease_seconds: int) -> dict[str, Any] | None:
    """Gives one waiting job to this worker.

    SKIP LOCKED lets workers pass over a row another worker is claiming, so two workers can never get
    the same job and none of them wait on each other. The oldest job per user goes first, so one
    person's big batch can't make everyone else wait behind it.
    """
    return connection.execute(
        """
        WITH running AS (
            SELECT user_id, count(*) AS n FROM jobs WHERE status = 'running' GROUP BY user_id
        ),
        next_job AS (
            SELECT j.id FROM jobs j
            LEFT JOIN running r ON r.user_id = j.user_id
            WHERE j.status = 'queued' AND j.run_after <= now()
            ORDER BY COALESCE(r.n, 0), j.created_at
            FOR UPDATE OF j SKIP LOCKED
            LIMIT 1
        )
        UPDATE jobs SET status = 'running', worker = %s, attempts = attempts + 1,
                        lease_until = now() + make_interval(secs => %s)
        FROM next_job WHERE jobs.id = next_job.id
        RETURNING jobs.*
        """,
        (worker, lease_seconds),
    ).fetchone()


def extend_lease(connection: Connection, job_id: uuid.UUID, worker: str, lease_seconds: int) -> bool:
    """Tells the queue this worker is still alive. Returns False if the job is no longer this worker's."""
    cursor = connection.execute(
        """UPDATE jobs SET lease_until = now() + make_interval(secs => %s)
           WHERE id = %s AND worker = %s AND status = 'running'""",
        (lease_seconds, job_id, worker),
    )
    return cursor.rowcount == 1


def requeue_expired_jobs(connection: Connection, max_attempts: int) -> int:
    """Recovers jobs whose worker died. They run again, or fail once out of attempts."""
    connection.execute(
        """UPDATE jobs SET status = 'failed', error = 'The worker stopped repeatedly on this file.',
                           finished_at = now(), lease_until = NULL
           WHERE status = 'running' AND lease_until < now() AND attempts >= %s""",
        (max_attempts,),
    )
    cursor = connection.execute(
        """UPDATE jobs SET status = 'queued', lease_until = NULL, worker = NULL
           WHERE status = 'running' AND lease_until < now()"""
    )
    return cursor.rowcount


def complete_job(connection: Connection, job_id: uuid.UUID, worker: str, machine: dict[str, Any]) -> bool:
    """Saves a result. A worker that lost its lease can't overwrite the newer attempt's work."""
    cursor = connection.execute(
        """UPDATE jobs SET status = %s, result = %s, finished_at = now(), lease_until = NULL, error = NULL
           WHERE id = %s AND worker = %s AND status = 'running'""",
        (machine["status"], Jsonb({"machine": machine}), job_id, worker),
    )
    return cursor.rowcount == 1


def fail_or_retry_job(
    connection: Connection, job_id: uuid.UUID, worker: str, error: str, max_attempts: int, retry: bool
) -> None:
    """Retries a temporary failure after a growing delay. A permanent one fails straight away."""
    connection.execute(
        """UPDATE jobs SET
             status = CASE WHEN %(retry)s AND attempts < %(max)s THEN 'queued' ELSE 'failed' END,
             run_after = now() + make_interval(secs => 15 * attempts * attempts),
             finished_at = CASE WHEN %(retry)s AND attempts < %(max)s THEN NULL ELSE now() END,
             error = %(error)s, lease_until = NULL, worker = NULL
           WHERE id = %(id)s AND worker = %(worker)s AND status = 'running'""",
        {"retry": retry, "max": max_attempts, "error": error[:500], "id": job_id, "worker": worker},
    )


def list_jobs(connection: Connection, user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    return connection.execute(
        """SELECT id, filename, status, page_count, attempts, error, created_at, finished_at,
                  reused_from IS NOT NULL AS reused
           FROM jobs WHERE user_id = %s ORDER BY created_at DESC LIMIT %s""",
        (user_id, limit),
    ).fetchall()


def get_job(connection: Connection, job_id: uuid.UUID, user_id: int) -> dict[str, Any] | None:
    """Every read filters by owner, so one user's job ID is useless to another user."""
    return connection.execute("SELECT * FROM jobs WHERE id = %s AND user_id = %s", (job_id, user_id)).fetchone()


def delete_job(connection: Connection, job_id: uuid.UUID, user_id: int) -> bool:
    cursor = connection.execute(
        "DELETE FROM jobs WHERE id = %s AND user_id = %s AND status <> 'running'", (job_id, user_id)
    )
    return cursor.rowcount == 1


def save_review(
    connection: Connection,
    job_id: uuid.UUID,
    user_id: int,
    reviewed: dict[str, Any],
    changes: list[tuple[str, Any, Any]],
    complete: bool,
) -> bool:
    """Saves corrections and their audit rows together, so neither can exist without the other."""
    with connection.transaction():
        cursor = connection.execute(
            """UPDATE jobs SET result = jsonb_set(result, '{reviewed}', %s, true),
                               status = CASE WHEN %s THEN 'reviewed' ELSE status END
               WHERE id = %s AND user_id = %s AND status IN ('accepted', 'needs_review', 'reviewed')""",
            (Jsonb(reviewed), complete, job_id, user_id),
        )
        if cursor.rowcount != 1:
            return False
        for field, old_value, new_value in changes:
            connection.execute(
                "INSERT INTO review_log (job_id, user_id, field, old_value, new_value) VALUES (%s, %s, %s, %s, %s)",
                (job_id, user_id, field, Jsonb(old_value), Jsonb(new_value)),
            )
    return True


def review_history(connection: Connection, job_id: uuid.UUID, user_id: int) -> list[dict[str, Any]]:
    return connection.execute(
        """SELECT l.field, l.old_value, l.new_value, l.at, u.username
           FROM review_log l
           JOIN jobs j ON j.id = l.job_id
           JOIN users u ON u.id = l.user_id
           WHERE l.job_id = %s AND j.user_id = %s ORDER BY l.at""",
        (job_id, user_id),
    ).fetchall()


def to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        json.dumps(row, default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value))
    )
    return result
