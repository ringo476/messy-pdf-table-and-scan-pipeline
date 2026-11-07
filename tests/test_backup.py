"""Backups must be checked after they are written, and old ones removed together with their pair."""

import gzip
import io
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pytest

from pdfpipe import backup

FAKE_COMPOSE = """
import gzip, io, os, sys, tarfile

mode = os.environ.get("FAKE_MODE", "good")
args = sys.argv[1:]
if "pg_dump" in args:
    if mode == "dump_fails":
        sys.stderr.write("pg_dump: connection refused")
        sys.exit(1)
    text = "-- PostgreSQL database dump\\nCREATE TABLE a (id int);\\n-- PostgreSQL database dump complete\\n"
    if mode == "dump_cut_off":
        text = "-- PostgreSQL database dump\\nCREATE TABLE a (id int);\\n"
    sys.stdout.buffer.write(text.encode())
elif "tar" in args:
    if mode == "tar_fails":
        sys.stderr.write("tar: no such file")
        sys.exit(2)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"%PDF-1.4 test"
        info = tarfile.TarInfo("jobs/one/input.pdf")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    payload = buffer.getvalue()
    if mode == "tar_cut_off":
        payload = payload[: len(payload) // 2]
    sys.stdout.buffer.write(payload)
"""


@pytest.fixture
def compose(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_compose.py"
    script.write_text(FAKE_COMPOSE, encoding="utf-8")
    return [sys.executable, str(script)]


NOW = datetime(2026, 11, 7, 12, 30, 45)


def test_a_good_backup_writes_two_checked_files(tmp_path: Path, compose: list[str]) -> None:
    dump, files = backup.make_backup(tmp_path / "out", compose, NOW)
    assert dump.name == "pdfpipe-20261107-123045-database.sql.gz"
    assert files.name == "pdfpipe-20261107-123045-files.tar.gz"
    with gzip.open(dump, "rt") as handle:
        assert "CREATE TABLE a" in handle.read()
    with tarfile.open(files) as archive:
        assert archive.getnames() == ["jobs/one/input.pdf"]


@pytest.mark.parametrize("mode", ["dump_fails", "dump_cut_off", "tar_fails", "tar_cut_off"])
def test_a_bad_backup_is_reported_and_leaves_nothing_behind(
    tmp_path: Path, compose: list[str], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(backup.BackupError):
        backup.make_backup(tmp_path / "out", compose, NOW)
    # A backup with one good half would look complete, so no half is kept, and no partial file is left.
    assert list((tmp_path / "out").iterdir()) == []


def test_a_failing_command_names_what_went_wrong(
    tmp_path: Path, compose: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "dump_fails")
    with pytest.raises(backup.BackupError, match="connection refused"):
        backup.make_backup(tmp_path / "out", compose, NOW)


class TestVerification:
    def test_a_file_that_is_not_gzip_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "x-database.sql.gz"
        path.write_bytes(b"this is not gzip")
        with pytest.raises(backup.BackupError):
            backup.verify_dump(path)

    def test_a_dump_without_its_closing_line_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "x-database.sql.gz"
        with gzip.open(path, "wt") as handle:
            handle.write("-- PostgreSQL database dump\nCREATE TABLE a (id int);\n")
        with pytest.raises(backup.BackupError, match="complete"):
            backup.verify_dump(path)

    def test_an_empty_archive_is_still_a_valid_archive(self, tmp_path: Path) -> None:
        path = tmp_path / "x-files.tar.gz"
        with tarfile.open(path, "w:gz"):
            pass
        backup.verify_archive(path)  # A system with no uploads yet has an empty data folder.

    def test_a_truncated_archive_is_refused(self, tmp_path: Path) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            data = os.urandom(50_000)  # random bytes don't compress, so cutting the file really cuts the data
            info = tarfile.TarInfo("big.bin")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        path = tmp_path / "x-files.tar.gz"
        path.write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 2])
        with pytest.raises(backup.BackupError):
            backup.verify_archive(path)


class TestPruning:
    def make(self, folder: Path, stamps: list[str]) -> None:
        folder.mkdir(exist_ok=True)
        for stamp in stamps:
            (folder / f"pdfpipe-{stamp}-database.sql.gz").write_bytes(b"x")
            (folder / f"pdfpipe-{stamp}-files.tar.gz").write_bytes(b"x")

    def test_the_newest_backups_are_kept_as_pairs(self, tmp_path: Path) -> None:
        self.make(tmp_path, ["20261101-010000", "20261102-010000", "20261103-010000", "20261104-010000"])
        removed = backup.prune(tmp_path, keep=2)
        assert len(removed) == 4  # two backups, two files each
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "pdfpipe-20261103-010000-database.sql.gz",
            "pdfpipe-20261103-010000-files.tar.gz",
            "pdfpipe-20261104-010000-database.sql.gz",
            "pdfpipe-20261104-010000-files.tar.gz",
        ]

    def test_nothing_is_removed_when_there_are_few_enough(self, tmp_path: Path) -> None:
        self.make(tmp_path, ["20261101-010000"])
        assert backup.prune(tmp_path, keep=14) == []

    def test_files_that_are_not_backups_are_never_touched(self, tmp_path: Path) -> None:
        self.make(tmp_path, ["20261101-010000", "20261102-010000"])
        (tmp_path / "notes.txt").write_text("keep me")
        (tmp_path / "pdfpipe-notes.sql.gz").write_text("keep me too")
        backup.prune(tmp_path, keep=1)
        assert (tmp_path / "notes.txt").exists()
        assert (tmp_path / "pdfpipe-notes.sql.gz").exists()

    def test_keep_is_never_zero_so_the_last_backup_cannot_be_pruned_away(self, tmp_path: Path) -> None:
        self.make(tmp_path, ["20261101-010000"])
        assert backup.prune(tmp_path, keep=0) == []
        assert len(list(tmp_path.iterdir())) == 2
