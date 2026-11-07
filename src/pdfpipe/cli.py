"""Command line: process one PDF, or create a login for the web app."""

import argparse
import getpass
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .engines import Engines
from .fields.llm import build_resolver
from .pipeline.intake import RejectedFile
from .process import process_pdf, result_to_dict


def _extract(args: argparse.Namespace) -> int:
    settings = Settings()
    output_dir = args.out
    try:
        result = process_pdf(args.pdf, settings, Engines(settings), output_dir, build_resolver(settings))
    except RejectedFile as error:
        print(f"Rejected: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result_to_dict(result), indent=2, ensure_ascii=False)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(text, encoding="utf-8")
        print(f"{result.status}: {output_dir / 'result.json'}")
    else:
        print(text)
    return 0


def _create_user(args: argparse.Namespace) -> int:
    from .service import auth, db

    password = getpass.getpass("Password (at least 10 characters): ")
    if len(password) < 10:
        print("The password is too short.", file=sys.stderr)
        return 2
    settings = Settings()
    with db.connect(settings) as connection:
        db.apply_schema(connection)
        auth.create_user(connection, args.username, password)
    print(f"Created user {args.username}")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="pdfpipe")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="process one PDF")
    extract.add_argument("pdf", type=Path)
    extract.add_argument("--out", type=Path, help="folder for result.json and chart images")
    extract.set_defaults(handler=_extract)
    create_user = commands.add_parser("create-user", help="create a login for the web app")
    create_user.add_argument("username")
    create_user.set_defaults(handler=_create_user)
    args = parser.parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
