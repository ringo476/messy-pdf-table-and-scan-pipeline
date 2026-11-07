"""A bad file must only cost the person who uploaded it. These run real child processes."""

import os
import threading
import time
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from pdfpipe.config import Settings
from pdfpipe.pipeline.intake import IntakeInfo, RejectedFile
from pdfpipe.service.safe_intake import inspect_in_process


def crash_like_a_segfault(path: Path, settings: Settings) -> IntakeInfo:
    os.abort()


def hang_forever(path: Path, settings: Settings) -> IntakeInfo:
    time.sleep(600)
    raise AssertionError("unreachable")


def make_pdf(path: Path) -> Path:
    page = canvas.Canvas(str(path))
    page.drawString(100, 100, "hello")
    page.save()
    return path


def test_a_good_file_passes(tmp_path: Path, settings: Settings) -> None:
    assert inspect_in_process(make_pdf(tmp_path / "a.pdf"), settings).page_count == 1


def test_a_bad_file_gets_the_normal_message(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(RejectedFile, match="not a PDF"):
        inspect_in_process(path, settings)


def test_a_library_crash_only_rejects_that_file(tmp_path: Path, settings: Settings) -> None:
    path = make_pdf(tmp_path / "a.pdf")
    with pytest.raises(RejectedFile, match="damaged"):
        inspect_in_process(path, settings, check=crash_like_a_segfault)
    assert inspect_in_process(path, settings).page_count == 1


def test_a_hang_is_stopped_at_the_time_limit(tmp_path: Path, settings: Settings) -> None:
    path = make_pdf(tmp_path / "a.pdf")
    started = time.monotonic()
    with pytest.raises(RejectedFile, match="too long"):
        inspect_in_process(path, settings, check=hang_forever, timeout_seconds=3)
    assert time.monotonic() - started < 20


def test_many_uploads_at_once_do_not_crash(tmp_path: Path, settings: Settings) -> None:
    """The regression test for the crash: PDFium called from several threads at the same time."""
    paths = [make_pdf(tmp_path / f"{index}.pdf") for index in range(8)]
    results: list[int] = []
    errors: list[BaseException] = []

    def check(path: Path) -> None:
        try:
            results.append(inspect_in_process(path, settings).page_count)
        except BaseException as error:  # noqa: BLE001 - The test reports any failure from the thread.
            errors.append(error)

    threads = [threading.Thread(target=check, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert results == [1] * 8
