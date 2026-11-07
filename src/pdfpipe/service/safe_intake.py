"""Runs the upload check in a separate process for each file.

This solves two problems. PDFium isn't thread-safe, and simultaneous uploads from different users crashed
the whole web server with a segmentation fault. A hostile PDF could also crash or hang the PDF library, and
that must only cost the person who uploaded it.
"""

import multiprocessing as mp
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Settings
from ..pipeline.intake import IntakeInfo, RejectedFile, inspect_pdf

_CONTEXT = mp.get_context("spawn")
# Starting a process costs memory, so a burst of uploads queues here rather than exhausting the PC.
_SLOTS = threading.BoundedSemaphore(4)
_TIMEOUT_SECONDS = 60.0


def _child(pipe: Any, check: Callable[[Path, Settings], IntakeInfo], path: Path, settings: Settings) -> None:
    try:
        pipe.send(("ok", check(path, settings)))
    except RejectedFile as error:
        pipe.send(("rejected", str(error)))
    except Exception:  # Anything unexpected means the file can't be read safely.
        pipe.send(("rejected", "The PDF is damaged and can't be opened."))
    finally:
        pipe.close()


def inspect_in_process(
    path: Path,
    settings: Settings,
    check: Callable[[Path, Settings], IntakeInfo] = inspect_pdf,
    timeout_seconds: float = _TIMEOUT_SECONDS,
) -> IntakeInfo:
    """Checks a PDF in its own process. A crash or hang rejects that file and affects nothing else."""
    with _SLOTS:
        parent_end, child_end = _CONTEXT.Pipe(duplex=False)
        process = _CONTEXT.Process(target=_child, args=(child_end, check, path, settings), daemon=True)
        process.start()
        child_end.close()
        try:
            if not parent_end.poll(timeout_seconds):
                raise RejectedFile("The PDF took too long to read, so it was rejected.")
            try:
                status, value = parent_end.recv()
            except EOFError as error:
                # The process died without replying, which means the file crashed the PDF library.
                raise RejectedFile("The PDF is damaged and can't be opened.") from error
        finally:
            parent_end.close()
            if process.is_alive():
                process.kill()
            process.join(5)
    if status == "rejected":
        raise RejectedFile(str(value))
    info: IntakeInfo = value
    return info
