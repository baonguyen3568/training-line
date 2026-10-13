from __future__ import annotations

import json
import secrets
import sqlite3
import string
from dataclasses import asdict
from pathlib import Path

import pytest

from vlm_pipeline.app.secret_store import (
    AUDIT_REL,
    DB_REL,
    KEY_REL,
    SecretKeyError,
    SecretStore,
    SecretUnreadableError,
    key_path,
    last4,
    load_key,
)

WHO = {"actor": "sess-abcdef123456789", "client": "127.0.0.1"}


def _store(tmp_path: Path) -> SecretStore:
    keydir = tmp_path / "cfg"
    keydir.mkdir(mode=0o700)
    return SecretStore(tmp_path / "proj", key_file=keydir / "secrets.key")


def _token() -> str:
    return "hf_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(34))


def test_round_trip_and_status_never_holds_the_value(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(20):  # random tokens: no response shape may ever carry one
        token = _token()
        status = store.set("HF_TOKEN", token, **WHO)
        assert store.reveal("HF_TOKEN") == token
        shown = json.dumps(asdict(store.status("HF_TOKEN")))
        assert token not in shown
        assert token[:-4] not in shown
        assert (status.set, status.last4, status.readable) == (True, token[-4:], True)


def test_the_database_never_holds_plaintext(tmp_path: Path) -> None:
    store = _store(tmp_path)
    token = _token()
    store.set("HF_TOKEN", token, **WHO)
    store.close()
    db = tmp_path / "proj" / DB_REL
    dump = "\n".join(sqlite3.connect(db).iterdump())
    assert token not in dump
    assert token.encode() not in db.read_bytes()
    assert db.stat().st_mode & 0o077 == 0


def test_a_ciphertext_moved_to_another_field_does_not_decrypt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("HF_TOKEN", _token(), **WHO)
    store.set("NGC_API_KEY", "nvapi-" + "x" * 30, **WHO)
    db = sqlite3.connect(tmp_path / "proj" / DB_REL)
    nonce, ct = db.execute("SELECT nonce, ciphertext FROM secrets WHERE field = 'HF_TOKEN'").fetchone()
    db.execute("UPDATE secrets SET nonce = ?, ciphertext = ? WHERE field = 'NGC_API_KEY'", (nonce, ct))
    db.commit()
    with pytest.raises(SecretUnreadableError):
        store.reveal("NGC_API_KEY")


def test_a_row_under_another_key_asks_to_enter_it_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("HF_TOKEN", _token(), **WHO)
    store.key_file.unlink()
    store.key_file.write_bytes(secrets.token_bytes(32))
    store.key_file.chmod(0o600)
    assert store.status("HF_TOKEN").readable is False
    with pytest.raises(SecretUnreadableError, match="enter it again"):
        store.reveal("HF_TOKEN")


def test_the_key_is_created_once_0600_in_a_private_folder(tmp_path: Path) -> None:
    path = tmp_path / "new" / "secrets.key"
    key = load_key(path, create=True)
    assert key is not None
    assert len(key) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert load_key(path, create=True) == key  # never replaced
    assert load_key(tmp_path / "absent.key") is None


@pytest.mark.parametrize("problem", ["0644", "symlink", "short", "group_dir"])
def test_an_unsafe_key_is_refused_and_never_regenerated(tmp_path: Path, problem: str) -> None:
    folder = tmp_path / "cfg"
    folder.mkdir(mode=0o700)
    path = folder / "secrets.key"
    real = secrets.token_bytes(32)
    if problem == "symlink":
        target = tmp_path / "elsewhere.key"
        target.write_bytes(real)
        target.chmod(0o600)
        path.symlink_to(target)
    else:
        path.write_bytes(real[:16] if problem == "short" else real)
        path.chmod(0o644 if problem == "0644" else 0o600)
        if problem == "group_dir":
            folder.chmod(0o770)
    before = path.read_bytes()
    with pytest.raises(SecretKeyError):
        load_key(path, create=True)
    assert path.read_bytes() == before  # an unusable key is an error, not a reason to make a new one


def test_audit_has_a_line_per_change_and_never_a_value(tmp_path: Path) -> None:
    store = _store(tmp_path)
    token = _token()
    store.set("HF_TOKEN", token, **WHO)
    store.delete("HF_TOKEN", **WHO)
    with pytest.raises(ValueError, match="characters"):
        store.set("HF_TOKEN", " padded ", **WHO)
    audit = tmp_path / "proj" / AUDIT_REL
    lines = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [(r["field"], r["action"]) for r in lines] == [
        ("HF_TOKEN", "set"),
        ("HF_TOKEN", "delete"),
        ("HF_TOKEN", "rejected"),
    ]
    assert lines[0]["last4"] == token[-4:]
    assert lines[0]["actor"] == WHO["actor"][:12]
    assert token not in audit.read_text()
    assert "padded" not in audit.read_text()
    assert store.status("HF_TOKEN").set is False


def test_only_known_secret_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.status("deployment.profile")  # admin policy is never a writable/readable setting here
    with pytest.raises(KeyError):
        store.set("VLM_PORT", "1", **WHO)


def test_without_key_creation_a_missing_key_is_an_error(tmp_path: Path) -> None:
    store = SecretStore(tmp_path / "proj", key_file=tmp_path / "none" / "secrets.key", create_key=False)
    with pytest.raises(SecretKeyError, match="installer"):
        store.set("HF_TOKEN", _token(), **WHO)
    assert not (tmp_path / "none").exists()


def test_short_values_show_no_tail() -> None:
    assert last4("abcdefgh") == ""
    assert last4("hf_abcdefghij1234") == "1234"


def test_key_path_ignores_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not key_path().is_relative_to(tmp_path)
    assert key_path().parts[-3:] == KEY_REL.parts
