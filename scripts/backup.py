"""Backs up the running system: the database and the uploaded files.

Run it on the PC where Docker is running, from the project folder:
    python scripts/backup.py
It writes two files per backup into the backups folder and keeps the newest 14. To restore, see the README.
"""

import argparse
import sys
from pathlib import Path

from pdfpipe.backup import BackupError, make_backup, prune


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up the database and the uploaded files.")
    parser.add_argument("--out", type=Path, default=Path("backups"), help="where to put the backups")
    parser.add_argument("--keep", type=int, default=14, help="how many backups to keep")
    args = parser.parse_args()
    try:
        dump, files = make_backup(args.out, ["docker", "compose"])
    except BackupError as error:
        print(f"Backup FAILED: {error}", file=sys.stderr)
        return 1
    removed = prune(args.out, args.keep)
    print(f"Backup written and checked: {dump.name} ({dump.stat().st_size:,} bytes)")
    print(f"Backup written and checked: {files.name} ({files.stat().st_size:,} bytes)")
    if removed:
        print(f"Removed {len(removed)} old backup files, keeping the newest {args.keep}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
