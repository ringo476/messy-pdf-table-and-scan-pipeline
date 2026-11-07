"""The retention rule: remove old uploads, and optionally old records, after the time the operator chose.

Both limits are off by default (0 days). Many businesses have to keep invoices for years, so nothing is removed
unless someone turns this on. A document that is still waiting for review is never touched, however old.

Two separate limits, because they remove different things:
- Files: the uploaded PDF and the chart images. The extracted result, the corrections, and the audit trail stay.
- Records: everything about the document, including its result and its audit trail.
"""

import logging
import shutil

from ..config import Settings
from . import db
from .paths import job_dir

logger = logging.getLogger("pdfpipe.retention")


def run_retention(connection: db.Connection, settings: Settings) -> tuple[int, int]:
    """Removes what has passed its limit. Returns (jobs whose files were removed, jobs removed entirely)."""
    files_removed = records_removed = 0

    if settings.retention_records_days > 0:
        expired = db.jobs_older_than(connection, settings.retention_records_days, only_with_files=False)
        if expired:
            # The files go first. If removing the rows then failed, the rows would still point at nothing harmful.
            for job_id in expired:
                shutil.rmtree(job_dir(settings, job_id), ignore_errors=True)
            records_removed = db.delete_jobs(connection, expired)

    if settings.retention_files_days > 0:
        expired = db.jobs_older_than(connection, settings.retention_files_days, only_with_files=True)
        if expired:
            for job_id in expired:
                shutil.rmtree(job_dir(settings, job_id), ignore_errors=True)
            db.mark_files_deleted(connection, expired)
            files_removed = len(expired)

    if files_removed or records_removed:
        logger.info("retention removed the files of %d jobs and %d whole jobs", files_removed, records_removed)
    return files_removed, records_removed
