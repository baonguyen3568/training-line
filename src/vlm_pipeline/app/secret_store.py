"""User secrets set from the Settings page (D-7, P015 §4.5/§4.6): encrypted at rest, write-only through the API.

* AES-256-GCM (``cryptography``, AESGCM only), a random 96-bit nonce per write, AAD = ``key_id:field`` so a
  ciphertext copied to another field (or encrypted under another key) never decrypts.
* The 32-byte key lives ONLY in ``~/.config/ntc-training/secrets.key`` (passwd home, no env override): never in the
  DB, a log, a job mount or a build context. v1: the app creates it on first use (``O_CREAT|O_EXCL|O_NOFOLLOW``,
  0600, folder 0700); in compose the host installer creates it and binds that one file read-only.
* A key file that fails a check (symlink, not a regular file, foreign owner, readable by group/others, wrong size)
  is an error: secrets are unusable, and a new key is NEVER generated over it (that would silently lose them all).
* Rows carry ``key_id`` (first 16 hex of the key's SHA-256): a row under another key reads as "re-enter the token".
* ``status`` is all the API ever returns: set / updated_at / last4 (4 chars at most, none for a short value).
* Every change appends ``outputs/system/settings_audit.jsonl``: never a secret value, only the new ``last4``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import secrets
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vlm_pipeline.common.app_config import APP_ID

KEY_REL = Path(".config") / APP_ID / "secrets.key"
DB_REL = Path("outputs") / "system" / "settings.db"
AUDIT_REL = Path("outputs") / "system" / "settings_audit.jsonl"
SECRET_FIELDS = frozenset({"HF_TOKEN", "NGC_API_KEY", "WANDB_API_KEY"})
KEY_BYTES = 32
MAX_SECRET_CHARS = 4096
LAST4_MIN_CHARS = 12  # a shorter value shows no tail at all (4 chars of 8 would be half of it)

Action = Literal["set", "delete", "rejected"]


class SecretKeyError(RuntimeError):
    """The key file exists but is unsafe or unreadable: no secret can be used until an admin fixes it."""


class SecretUnreadableError(RuntimeError):
    """The stored value cannot be decrypted (other key, tampered row): the user must enter it again."""


def key_path() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / KEY_REL


def _check_dir(path: Path) -> None:
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise SecretKeyError(f"{path} must be a directory owned by this user, not writable by group/others")


def _read_key(fd: int, path: Path) -> bytes:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SecretKeyError(f"{path} must be a regular file owned by this user with mode 0600")
    key = os.read(fd, KEY_BYTES + 1)
    if len(key) != KEY_BYTES:
        raise SecretKeyError(f"{path} must hold exactly {KEY_BYTES} bytes")
    return key


def load_key(path: Path | None = None, *, create: bool = False) -> bytes | None:
    """The key, or None when absent and ``create`` is False. Raises SecretKeyError for an unsafe file."""
    path = path if path is not None else key_path()
    if path.parent.exists():
        _check_dir(path.parent)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        if not create:
            return None
        return _create_key(path)
    except OSError as exc:  # ELOOP: a symlink
        raise SecretKeyError(f"{path} is unusable ({exc.strerror})") from None
    try:
        return _read_key(fd, path)
    finally:
        os.close(fd)


def _create_key(path: Path) -> bytes:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_dir(path.parent)
    key = secrets.token_bytes(KEY_BYTES)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, key)
        os.fsync(fd)
    finally:
        os.close(fd)
    return key


def key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def last4(value: str) -> str:
    return value[-4:] if len(value) >= LAST4_MIN_CHARS else ""


@dataclass(frozen=True)
class SecretStatus:
    set: bool
    updated_at: float | None = None
    last4: str = ""
    readable: bool = True  # False: stored under another key or damaged -> "enter the token again"


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS secrets (field TEXT PRIMARY KEY, key_id TEXT NOT NULL, nonce BLOB NOT NULL,"
    " ciphertext BLOB NOT NULL, last4 TEXT NOT NULL, updated_at REAL NOT NULL)"
)


class SecretStore:
    """Encrypted secrets in ``outputs/system/settings.db``. The key is loaded lazily and only for set/reveal."""

    def __init__(self, root: Path, *, key_file: Path | None = None, create_key: bool = True) -> None:
        self.root = root
        self.key_file = key_file if key_file is not None else key_path()
        self.create_key = create_key
        db = root / DB_REL
        db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        os.close(os.open(db, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600))  # 0600 from the start
        self._db = sqlite3.connect(db, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.execute(_SCHEMA)

    def _key(self, *, create: bool) -> bytes | None:
        return load_key(self.key_file, create=create and self.create_key)

    @staticmethod
    def _aad(kid: str, field: str) -> bytes:
        return f"{kid}:{field}".encode()

    def set(self, field: str, value: str, *, actor: str, client: str) -> SecretStatus:
        _check_field(field)
        if not value or len(value) > MAX_SECRET_CHARS or value != value.strip() or not _printable_ascii(value):
            self._audit(field, "rejected", actor=actor, client=client)
            raise ValueError(
                f"{field}: value must be 1..{MAX_SECRET_CHARS} printable ASCII characters without surrounding spaces"
            )
        key = self._key(create=True)
        if key is None:  # only when create_key=False (compose: the installer made the key)
            raise SecretKeyError(f"{self.key_file} is missing: run the installer to create it")
        kid, nonce = key_id(key), secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, value.encode(), self._aad(kid, field))
        now, tail = time.time(), last4(value)
        with self._lock:
            self._db.execute(
                "INSERT INTO secrets VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(field) DO UPDATE SET key_id=excluded.key_id,"
                " nonce=excluded.nonce, ciphertext=excluded.ciphertext, last4=excluded.last4,"
                " updated_at=excluded.updated_at",
                (field, kid, nonce, ct, tail, now),
            )
        self._audit(field, "set", actor=actor, client=client, last4=tail)
        return SecretStatus(set=True, updated_at=now, last4=tail)

    def delete(self, field: str, *, actor: str, client: str) -> None:
        _check_field(field)
        with self._lock:
            self._db.execute("DELETE FROM secrets WHERE field = ?", (field,))
        self._audit(field, "delete", actor=actor, client=client)

    def status(self, field: str) -> SecretStatus:
        """What the API may show. Never the value; ``readable`` False when the row is under another key."""
        _check_field(field)
        with self._lock:
            row = self._db.execute("SELECT key_id, last4, updated_at FROM secrets WHERE field = ?", (field,)).fetchone()
        if row is None:
            return SecretStatus(set=False)
        try:
            key = self._key(create=False)
        except SecretKeyError:
            key = None
        return SecretStatus(
            set=True, updated_at=row[2], last4=row[1], readable=key is not None and key_id(key) == row[0]
        )

    def reveal(self, field: str) -> str | None:
        """Plaintext for the one consumer that needs it (the download job). Never for an API response or a log."""
        _check_field(field)
        with self._lock:
            row = self._db.execute("SELECT key_id, nonce, ciphertext FROM secrets WHERE field = ?", (field,)).fetchone()
        if row is None:
            return None
        key = self._key(create=False)
        if key is None or key_id(key) != row[0]:
            raise SecretUnreadableError(f"{field} was saved under another key: enter it again")
        try:
            return AESGCM(key).decrypt(row[1], row[2], self._aad(row[0], field)).decode()
        except InvalidTag:
            raise SecretUnreadableError(f"{field} cannot be decrypted: enter it again") from None

    def close(self) -> None:
        self._db.close()

    def _audit(self, field: str, action: Action, *, actor: str, client: str, last4: str | None = None) -> None:
        path = self.root / AUDIT_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": time.time(), "field": field, "action": action, "actor": actor[:12], "client": client}
        if last4:
            line["last4"] = last4
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _printable_ascii(value: str) -> bool:
    """Every HF/NGC/W&B token is printable ASCII; a control char would split a header or a secrets file line."""
    return value.isascii() and value.isprintable()


def _check_field(field: str) -> None:
    if field not in SECRET_FIELDS:
        raise KeyError(f"unknown secret field {field!r}")


__all__ = [
    "AUDIT_REL",
    "DB_REL",
    "KEY_REL",
    "SECRET_FIELDS",
    "SecretKeyError",
    "SecretStatus",
    "SecretStore",
    "SecretUnreadableError",
    "key_id",
    "key_path",
    "last4",
    "load_key",
]
