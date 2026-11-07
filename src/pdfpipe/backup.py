"""Backs up the database and the uploaded files from the running Docker stack, and removes old backups.

A backup is two files with the same timestamp: a compressed SQL dump of the database, and a compressed archive of
the data folder (uploaded PDFs and chart images). Each one is checked after it is written, because a backup that
can't be read back is worse than none.

This runs on the host, not inside a container, because it needs the Docker command.
"""

import gzip
import re
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

DUMP_SUFFIX = "-database.sql.gz"
FILES_SUFFIX = "-files.tar.gz"
_NAME = re.compile(r"^pdfpipe-(\d{8}-\d{6})(-database\.sql\.gz|-files\.tar\.gz)$")


class BackupError(Exception):
    """A backup couldn't be made or didn't check out. The message says which part failed."""


def backup_stem(now: datetime) -> str:
    return f"pdfpipe-{now:%Y%m%d-%H%M%S}"


def _run_to_file(command: list[str], target: Path, compress: bool) -> None:
    """Runs a command and writes what it prints to a file. A failed command leaves no half-written file behind."""
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            assert process.stdout is not None
            opener = gzip.open if compress else open
            with opener(partial, "wb") as out:
                shutil.copyfileobj(process.stdout, out)
            error = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            if process.wait() != 0:
                raise BackupError(f"{' '.join(command[:5])} failed: {error.strip()[:300]}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def verify_dump(path: Path) -> None:
    """The dump must be readable gzip and must look like a PostgreSQL dump that finished."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except (OSError, EOFError) as error:
        raise BackupError(f"{path.name} can't be read back: {error}") from error
    if "PostgreSQL database dump" not in text[:2000] or "PostgreSQL database dump complete" not in text[-2000:]:
        raise BackupError(f"{path.name} doesn't look like a complete database dump")


def verify_archive(path: Path) -> None:
    """The archive must open, and every member must be readable to the end."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        while extracted.read(1024 * 1024):
                            pass
    except (OSError, EOFError, tarfile.TarError) as error:
        raise BackupError(f"{path.name} can't be read back: {error}") from error


def make_backup(folder: Path, compose: list[str], now: datetime | None = None) -> tuple[Path, Path]:
    """Writes and checks one backup. compose is the docker compose command, for example ["docker", "compose"]."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = backup_stem(now or datetime.now())
    dump, files = folder / f"{stem}{DUMP_SUFFIX}", folder / f"{stem}{FILES_SUFFIX}"
    try:
        _run_to_file(
            [*compose, "exec", "-T", "db", "pg_dump", "-U", "pdfpipe", "-d", "pdfpipe", "--no-owner"], dump, True
        )
        verify_dump(dump)
        # The archive is already compressed by tar, so it is written as it comes.
        _run_to_file([*compose, "exec", "-T", "web", "tar", "-czf", "-", "-C", "/data", "."], files, False)
        verify_archive(files)
    except BackupError:
        # A backup with one good half and one bad half would look complete, so neither is kept.
        dump.unlink(missing_ok=True)
        files.unlink(missing_ok=True)
        raise
    return dump, files


def prune(folder: Path, keep: int) -> list[Path]:
    """Removes all but the newest `keep` backups. A backup is both of its files, so they are removed together."""
    stems: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        match = _NAME.match(path.name)
        if match:
            stems.setdefault(match.group(1), []).append(path)
    removed: list[Path] = []
    for stamp in sorted(stems, reverse=True)[max(keep, 1) :]:
        for path in stems[stamp]:
            path.unlink()
            removed.append(path)
    return removed
