"""Local web app hardening (Proposal 003 D13): Host/Origin allowlist, startup token, CSRF, WebSocket auth.

The app binds to 127.0.0.1 on a shared host, so "localhost only" is not an access control by itself: other users,
other services and ``--network host`` containers can reach the port, cookies are not isolated by port, and
SameSite treats every 127.0.0.1 port as the same site. Defences, in order:

1. **Host allowlist** (exact ``127.0.0.1:<port>`` / ``localhost:<port>``) on every request -> DNS rebinding.
2. **Startup token**: printed once as ``http://127.0.0.1:<port>/?token=...``; ``GET /?token=`` swaps it for an
   ``HttpOnly; SameSite=Strict`` session cookie and redirects so the token leaves the URL bar. No cookie -> 401.
3. **Origin** must match exactly (scheme + host + port) on every state-changing request and WebSocket handshake.
4. **CSRF token** = HMAC(per-process secret, session); fetched from ``GET /api/session`` and kept in memory by the
   client (never a readable cookie: another app on 127.0.0.1 could toss it). Required as ``X-VLM-CSRF`` on writes.
5. **WebSocket**: browsers cannot set headers, so after the handshake checks the first message must be
   ``{"type": "auth", "csrf_token": ...}`` (<= 4 KiB, within 5 s) or the socket is closed with 4401.

No CORS headers are ever emitted. Access is logged by this middleware to ``vlm_pipeline.access`` with the query
string redacted: run uvicorn with ``access_log=False`` so its own logger never sees ``?token=``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qs

ACCESS_LOGGER = "vlm_pipeline.access"
SESSION_COOKIE = "vlm_session"
CSRF_HEADER = "x-vlm-csrf"
SESSION_PATH = "/api/session"
DEV_PORT = 7875  # Vite dev server; its proxy keeps the browser's Host/Origin (changeOrigin: false)
WS_CLOSE_UNAUTHORIZED = 4401
WS_AUTH_MAX_BYTES = 4096
WS_AUTH_TIMEOUT_S = 5.0
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

access_log = logging.getLogger(ACCESS_LOGGER)


class _App(Protocol):
    state: Any

    def add_middleware(self, middleware_class: Any, **options: Any) -> None: ...


@dataclass(frozen=True)
class SecurityConfig:
    """Everything the middleware needs; one instance per process (new secret and token on every restart)."""

    token: str
    port: int
    dev: bool = False
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    ws_auth_timeout_s: float = WS_AUTH_TIMEOUT_S

    @property
    def allowed_hosts(self) -> frozenset[str]:
        ports = [self.port, DEV_PORT] if self.dev else [self.port]
        return frozenset(f"{h}:{p}" for p in ports for h in ("127.0.0.1", "localhost"))

    @property
    def allowed_origins(self) -> frozenset[str]:
        return frozenset(f"http://{h}" for h in self.allowed_hosts)

    def session_ok(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value.encode(), self.token.encode())

    def csrf_token(self) -> str:
        return hmac.new(self.secret, b"csrf:" + self.token.encode(), hashlib.sha256).hexdigest()

    def csrf_ok(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value.encode(), self.csrf_token().encode())


def _headers(scope: Scope) -> dict[str, str]:
    # Duplicate headers: keep the first; Host/Origin/Cookie duplicates are rejected below anyway.
    out: dict[str, str] = {}
    for raw_key, raw_value in scope.get("headers", []):
        key = raw_key.decode("latin-1").lower()
        out.setdefault(key, raw_value.decode("latin-1"))
    return out


def _duplicated(scope: Scope, names: Iterable[str]) -> bool:
    wanted = set(names)
    seen: set[str] = set()
    for raw_key, _ in scope.get("headers", []):
        key = raw_key.decode("latin-1").lower()
        if key in wanted:
            if key in seen:
                return True
            seen.add(key)
    return False


def _cookie(header: str | None, name: str) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value
    return None


def _redacted_path(scope: Scope) -> str:
    path = str(scope.get("path", ""))
    query = scope.get("query_string", b"").decode("latin-1")
    if not query:
        return path
    keys = [part.split("=", 1)[0] for part in query.split("&") if part]
    return path + "?" + "&".join(f"{k}=<redacted>" for k in keys)


async def _respond(send: Send, status: int, body: dict[str, object], extra: Iterable[tuple[bytes, bytes]] = ()) -> None:
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    headers += list(SECURITY_HEADERS) + list(extra)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})


class SecurityMiddleware:
    """Pure ASGI middleware (no framework dependency) enforcing D13 for HTTP and WebSocket scopes."""

    def __init__(self, app: ASGIApp, config: SecurityConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
        else:  # lifespan
            await self.app(scope, receive, send)

    # ------------------------------------------------------------------------------------------------- http
    async def _http(self, scope: Scope, receive: Receive, send: Send) -> None:
        status_holder = {"status": 0}

        async def send_logged(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                headers = [
                    (k, v) for k, v in message.get("headers", []) if not k.lower().startswith(b"access-control-")
                ]
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in SECURITY_HEADERS if k not in present]
                message["headers"] = headers
            await send(message)

        try:
            await self._http_checked(scope, receive, send_logged)
        finally:
            access_log.info("%s %s %s", scope.get("method", "?"), _redacted_path(scope), status_holder["status"])

    def _http_rejection(self, scope: Scope, headers: dict[str, str], method: str) -> tuple[int, str] | None:
        """``(status, code)`` when the request must be refused before reaching the app, else ``None``."""
        cfg = self.config
        dup = _duplicated(scope, ("host", "origin", "cookie", CSRF_HEADER))
        if dup or headers.get("host") not in cfg.allowed_hosts:
            return 403, "forbidden_host"
        if not cfg.session_ok(_cookie(headers.get("cookie"), SESSION_COOKIE)):
            return 401, "unauthenticated"
        if method in UNSAFE_METHODS and headers.get("origin") not in cfg.allowed_origins:
            return 403, "forbidden_origin"
        if method in UNSAFE_METHODS and not cfg.csrf_ok(headers.get(CSRF_HEADER)):
            return 403, "csrf_invalid"
        return None

    async def _bootstrap(self, send: Send, method: str, given: list[str]) -> None:
        """``GET /?token=<startup token>`` -> session cookie + redirect, so the token leaves the URL bar."""
        cfg = self.config
        if method != "GET" or len(given) != 1 or not cfg.session_ok(given[0]):
            await _respond(send, 401, {"code": "invalid_token"})
            return
        cookie = f"{SESSION_COOKIE}={cfg.token}; HttpOnly; SameSite=Strict; Path=/".encode()
        headers = [(b"location", b"/"), (b"set-cookie", cookie), (b"content-length", b"0")]
        await send({"type": "http.response.start", "status": 303, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    async def _http_checked(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = _headers(scope)
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        host_ok = headers.get("host") in self.config.allowed_hosts and not _duplicated(scope, ("host",))
        if host_ok and path == "/" and "token" in query:
            await self._bootstrap(send, method, query["token"])
        elif (rejection := self._http_rejection(scope, headers, method)) is not None:
            status, code = rejection
            body: dict[str, object] = {"code": code}
            if status == 401:
                body["hint"] = "open the URL printed in the terminal"
            await _respond(send, status, body)
        elif path == SESSION_PATH and method == "GET":
            await _respond(send, 200, {"csrf_token": self.config.csrf_token()})
        else:
            await self.app(scope, receive, send)

    # -------------------------------------------------------------------------------------------- websocket
    async def _websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        cfg = self.config
        headers = _headers(scope)
        ok = (
            not _duplicated(scope, ("host", "origin", "cookie"))
            and headers.get("host") in cfg.allowed_hosts
            and headers.get("origin") in cfg.allowed_origins
            and cfg.session_ok(_cookie(headers.get("cookie"), SESSION_COOKIE))
        )
        access_log.info("WS %s %s", _redacted_path(scope), "accepted-handshake" if ok else "rejected")
        if not ok:
            message = await receive()  # websocket.connect
            if message.get("type") == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})
            return
        scope.setdefault("state", {})["vlm_security"] = cfg
        await self.app(scope, receive, send)


class _WebSocketLike(Protocol):
    scope: Any

    async def accept(self) -> None: ...

    async def receive_text(self) -> str: ...

    async def close(self, code: int = ...) -> None: ...


async def ws_authenticate(websocket: _WebSocketLike) -> None:
    """Accept, then require ``{"type": "auth", "csrf_token": ...}`` as the first message; else close 4401 and raise.

    Call at the start of every WebSocket endpoint, before sending anything. Raises ``PermissionError`` after closing.
    """
    cfg: SecurityConfig | None = websocket.scope.get("state", {}).get("vlm_security")
    await websocket.accept()
    if cfg is None:  # middleware not installed: fail closed
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        raise PermissionError("websocket security middleware not installed")
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=cfg.ws_auth_timeout_s)
        if len(raw.encode()) > WS_AUTH_MAX_BYTES:
            raise ValueError("auth message too large")
        msg = json.loads(raw)
        if not (isinstance(msg, dict) and msg.get("type") == "auth" and cfg.csrf_ok(str(msg.get("csrf_token", "")))):
            raise ValueError("bad auth message")
    except (TimeoutError, ValueError, TypeError, KeyError) as exc:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        raise PermissionError("websocket authentication failed") from exc


def install_security(
    app: _App,
    *,
    port: int,
    dev: bool = False,
    token: str | None = None,
    secret: bytes | None = None,
    ws_auth_timeout_s: float = WS_AUTH_TIMEOUT_S,
) -> str:
    """Install the middleware on a FastAPI/Starlette app and return the startup token to print once.

    ``token``/``secret``/``ws_auth_timeout_s`` are for tests only; production lets this generate a fresh random
    token and secret per process (so a restart invalidates every session and CSRF token).
    """
    config = SecurityConfig(
        token=token or secrets.token_urlsafe(32),
        port=port,
        dev=dev,
        secret=secret or secrets.token_bytes(32),
        ws_auth_timeout_s=ws_auth_timeout_s,
    )
    app.add_middleware(SecurityMiddleware, config=config)
    app.state.vlm_security = config
    return config.token


__all__ = [
    "ACCESS_LOGGER",
    "SESSION_COOKIE",
    "WS_CLOSE_UNAUTHORIZED",
    "SecurityConfig",
    "SecurityMiddleware",
    "install_security",
    "ws_authenticate",
]
