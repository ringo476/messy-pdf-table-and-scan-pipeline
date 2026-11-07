"""Where a document's files live on disk."""

import uuid
from pathlib import Path

from ..config import Settings


def job_dir(settings: Settings, job_id: uuid.UUID | str) -> Path:
    return settings.data_dir / "jobs" / str(job_id)
