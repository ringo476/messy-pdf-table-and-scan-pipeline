"""Per-user login. Passwords use scrypt, and browser sessions use random tokens stored only as hashes."""

import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from .db import Connection

_N, _R, _P = 2**15, 8, 1
_MAX_MEMORY = 64 * 1024 * 1024
SESSION_COOKIE = "pdfpipe_session"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, maxmem=_MAX_MEMORY)
    return f"scrypt${_N}${_R}${_P}${base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, expected = stored.split("$")
        derived = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), maxmem=_MAX_MEMORY
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, base64.b64decode(expected))


# Checked when a username doesn't exist, so a login takes equally long either way.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def create_user(connection: Connection, username: str, password: str) -> int:
    row = connection.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (username, hash_password(password)),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def check_login(connection: Connection, username: str, password: str) -> int | None:
    row = connection.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,)).fetchone()
    if row is None:
        verify_password(password, _DUMMY_HASH)
        return None
    return int(row["id"]) if verify_password(password, row["password_hash"]) else None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(connection: Connection, user_id: int, hours: int) -> str:
    token = secrets.token_urlsafe(32)
    connection.execute("DELETE FROM sessions WHERE expires_at < now()")
    connection.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
        (_hash_token(token), user_id, datetime.now(UTC) + timedelta(hours=hours)),
    )
    return token


def user_for_session(connection: Connection, token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    return connection.execute(
        """SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = %s AND s.expires_at > now()""",
        (_hash_token(token),),
    ).fetchone()


def end_session(connection: Connection, token: str | None) -> None:
    if token:
        connection.execute("DELETE FROM sessions WHERE token_hash = %s", (_hash_token(token),))
