"""Settings page API (D-7, P015 §4.5): secrets are write-only, admin policy is read-only, the schema is closed.

* ``GET /api/settings``: per secret field ``{set, updated_at, last4, readable, scope}``, plus the admin policy values
  (read-only, for display) and whether the key file is usable. Never a secret value: this module cannot even reach
  one (``SecretStore.reveal`` is for the download job only; a test keeps it out of this file).
* ``PUT /api/settings/{key}`` ``{"value": "..."}`` saves a secret; ``DELETE /api/settings/{key}`` removes it.
  An admin-policy key -> 403 (only ``~/.config/ntc-training/policy.toml`` sets it), any other key -> 422.
* Behind ``SecurityMiddleware`` like every route (session cookie, Host/Origin, CSRF for PUT/DELETE); body <= 16 KiB,
  read with a cap before parsing; unknown body fields -> 422.
* Token scope (PM 2026-10-13): a token with write access is saved and used, with a one-time warning; until the
  runner can call ``whoami`` the scope is ``unchecked`` ("chưa kiểm được quyền token"), never a refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vlm_pipeline.app.secret_store import MAX_SECRET_CHARS, SECRET_FIELDS, SecretKeyError, SecretStore, load_key
from vlm_pipeline.common.app_config import effective_profile, read_policy

MAX_BODY_BYTES = 16 * 1024
PROFILE_ENV = "VLM_PROFILE"
ADMIN_POLICY_KEYS = (
    "deployment.profile",
    "storage.schedule",
    "storage.approval_policy",
    "storage.approval_threshold_gb",
    "plugins.allowlist",
)


def secret_value(raw: bytes) -> tuple[str | None, str | None]:  # noqa: PLR0911 - one return per rule
    """``(value, None)`` or ``(None, reason)``; the reason names the rule, never the input (it may be a token).

    Closed schema ``{"value": str}``; reasons: not_json | extra_field | not_text | empty | whitespace | not_printable
    | too_long.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return None, "not_json"
    if not isinstance(body, dict):
        return None, "not_json"
    if set(body) - {"value"}:
        return None, "extra_field"
    value = body.get("value")
    if not isinstance(value, str):
        return None, "not_text"
    if not value:
        return None, "empty"
    if value != value.strip():
        return None, "whitespace"
    if not (value.isascii() and value.isprintable()):  # \n \r \x00 bidi...: header/line injection downstream
        return None, "not_printable"
    if len(value) > MAX_SECRET_CHARS:
        return None, "too_long"
    return value, None


def _error(status: int, code: str, **params: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "params": params}})


def _actor(request: Request) -> str:
    """A short, stable id of the browser session for the audit: a hash, never the cookie itself."""
    sid = getattr(request.state, "vlm_sid", None)  # set by the security middleware (the session id, never a token)
    return hashlib.sha256(sid.encode()).hexdigest()[:12] if isinstance(sid, str) else "local"


def _admin_values(policy: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Each admin key as ``{value, source}``: ``deployment`` when policy.toml sets it, else ``default``.

    A ``default`` value is null (the app's built-in default applies). No ``reason_key`` yet: policy.toml has no
    field for one.
    """
    out: dict[str, dict[str, Any]] = {}
    for dotted in ADMIN_POLICY_KEYS:
        section, _, name = dotted.partition(".")
        table = (policy or {}).get(section)
        value = table.get(name) if isinstance(table, dict) else None
        out[dotted] = {"value": value, "source": "deployment" if value is not None else "default"}
    if out["storage.approval_policy"]["value"] == "none" and _shared_host(policy):
        out["storage.approval_policy"]["warning"] = "APPROVAL_OFF_SHARED"  # allowed, never silent (security)
    return out


def _shared_host(policy: dict[str, Any] | None) -> bool:
    """Not an effective ``personal`` install (same rule as storage: no profile anywhere counts as shared)."""
    try:
        return effective_profile(os.environ.get(PROFILE_ENV), policy) != "personal"
    except ValueError:  # an unknown VLM_PROFILE: treat as shared (fail closed)
        return True


async def _read_body(request: Request) -> bytes | None:
    """The body, or None once it passes MAX_BODY_BYTES (counted while reading: Content-Length is not trusted)."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def register_settings_routes(
    app: FastAPI, store: SecretStore, *, policy: Callable[[], dict[str, Any] | None] = read_policy
) -> None:
    def refused(key: str) -> JSONResponse | None:
        if key in ADMIN_POLICY_KEYS:
            return _error(403, "settings_admin_only", key=key)
        if key not in SECRET_FIELDS:
            return _error(422, "settings_unknown_key", key=key)
        return None

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        try:
            key_ok = load_key(store.key_file) is not None or store.create_key
        except SecretKeyError:
            key_ok = False
        fields = {field: {**asdict(store.status(field)), "scope": "unchecked"} for field in sorted(SECRET_FIELDS)}
        return {"secrets": fields, "admin_policy": _admin_values(policy()), "key_ok": key_ok}

    @app.put("/api/settings/{key}")
    async def put_setting(key: str, request: Request) -> JSONResponse:
        if (bad := refused(key)) is not None:
            return bad
        raw = await _read_body(request)
        if raw is None:
            return _error(413, "settings_body_too_large", limit=MAX_BODY_BYTES)
        value, reason = secret_value(raw)
        if value is None:
            return _error(422, "settings_invalid", key=key, reason=reason)
        client = request.client.host if request.client else "?"
        try:
            status = store.set(key, value, actor=_actor(request), client=client)
        except SecretKeyError:
            return _error(409, "settings_key_unusable")
        except ValueError:
            return _error(422, "settings_invalid", key=key, reason="rejected")
        return JSONResponse({**asdict(status), "scope": "unchecked"})

    @app.delete("/api/settings/{key}")
    def delete_setting(key: str, request: Request) -> JSONResponse:
        if (bad := refused(key)) is not None:
            return bad
        client = request.client.host if request.client else "?"
        store.delete(key, actor=_actor(request), client=client)
        return JSONResponse({"set": False})


__all__ = ["ADMIN_POLICY_KEYS", "MAX_BODY_BYTES", "register_settings_routes"]
