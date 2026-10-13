"""Web app hardening (Proposal 003 D13, ADR-019 LAN access): Host/Origin allowlist, login, sessions, CSRF, WS auth.

The app binds to 127.0.0.1 on a shared host (or, in compose, sits behind the host's 127.0.0.1 port mapping and the
LAN nginx). "Local only" is not an access control by itself: other users, other services and ``--network host``
containers reach the port, cookies are not isolated by port, and SameSite treats every 127.0.0.1 port as one site.
Defences, in order:

1. **Host allowlist** on every request -> DNS rebinding. Two classes, decided by the Host header ONLY:
   - *local*: exactly ``127.0.0.1:<port>`` / ``localhost:<port>``;
   - *lan*: exactly the netloc of one of ``VLM_PUBLIC_ORIGINS`` (``https://<ip|name>[:port]``, reached via nginx).
   ``X-Forwarded-*`` is never read: scheme, cookie and Origin all follow from the allowlisted Host. nginx forwards
   ``Host $host`` after ``server_name`` filtering (default_server 444), so a LAN client cannot claim a local Host
   through it; when ``proxy_peer`` (nginx's fixed compose IP) is set, a local Host from that peer is refused too,
   and a LAN Host from anything else. A local client faking a LAN Host only gets the stricter rules.
2. **Login**: a startup token (random, new on every restart, rotated by "sign out everywhere"). The printed link
   carries it in the fragment (``/#t=...``): the browser never sends it; the small login page served to an
   unauthenticated browser POSTs it to ``/api/session`` (JSON, exact Origin). ``GET /?token=`` still works but only
   for a local Host. Failed LAN logins are counted: a warning after 20 per 10 min, LAN login locked 15 min after 200
   (nginx limits 5/min per IP first; local login is never locked, so a LAN machine cannot lock the owner out).
3. **Sessions**: a random 256-bit id in an ``HttpOnly; SameSite=Strict`` cookie (``__Host-vlm_session``, Secure,
   for the LAN class; ``vlm_session`` for local), kept in memory with the class it was created for (12 h idle,
   7 days absolute, at most 64: the oldest goes). Each Host class reads only its own cookie name, and a session is
   refused under the other class even if its value is copied.
4. **Origin** must equal the Host's own origin on every state-changing request and WebSocket handshake.
5. **CSRF token** = HMAC(per-process secret, session id); fetched from ``GET /api/session``, kept in memory by the
   client (never a readable cookie). Required as ``X-VLM-CSRF`` on writes.
6. **WebSocket**: browsers cannot set headers, so after the handshake checks the first message must be
   ``{"type": "auth", "csrf_token": ...}`` (<= 4 KiB, within 5 s) or the socket is closed with 4401. An open socket
   is closed with 4401 when its session ends (sign out, sign out everywhere, expiry checked every minute).

No CORS headers are ever emitted. Access is logged by this middleware to ``vlm_pipeline.access`` with the query
string redacted: run uvicorn with ``access_log=False`` so its own logger never sees ``?token=``.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import parse_qs

ACCESS_LOGGER = "vlm_pipeline.access"
SESSION_COOKIE = "vlm_session"  # local class (http://127.0.0.1)
LAN_SESSION_COOKIE = "__Host-vlm_session"  # LAN class (https via nginx): Secure, no Domain, Path=/
CSRF_HEADER = "x-vlm-csrf"
SESSION_PATH = "/api/session"
REVOKE_ALL_PATH = "/api/session/revoke-all"
UNLOCK_LAN_PATH = "/api/session/unlock-lan"
DEV_PORT = 4010  # Vite dev server (D-8); its proxy keeps the browser's Host/Origin (changeOrigin: false)
PUBLIC_ORIGINS_ENV = "VLM_PUBLIC_ORIGINS"
PROXY_PEER_ENV = "VLM_PROXY_PEER"
WS_CLOSE_UNAUTHORIZED = 4401
WS_AUTH_MAX_BYTES = 4096
WS_AUTH_TIMEOUT_S = 5.0
LOGIN_MAX_BYTES = 4096
WS_SESSION_CHECK_S = 60.0  # an open WebSocket re-checks its session (expiry) this often
SESSION_IDLE_S = 12 * 3600
SESSION_ABSOLUTE_S = 7 * 24 * 3600
MAX_SESSIONS = 64
LOGIN_WINDOW_S = 600
LOGIN_WARN_FAILURES = 20
LOGIN_LOCK_FAILURES = 200
LOGIN_LOCK_S = 900
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
)

HostClass = Literal["local", "lan"]
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

access_log = logging.getLogger(ACCESS_LOGGER)
log = logging.getLogger(__name__)
_now: Callable[[], float] = time.time  # tests replace it


class _App(Protocol):
    state: Any

    def add_middleware(self, middleware_class: Any, **options: Any) -> None: ...


# ------------------------------------------------------------------------------------------------ public origins
_PUBLIC_ORIGIN = re.compile(r"https://(?P<host>[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?)(?::(?P<port>[0-9]{1,5}))?")


def parse_public_origins(raw: str) -> tuple[str, ...]:
    """``VLM_PUBLIC_ORIGINS`` (comma separated ``https://<ipv4|name>[:port]``), validated; ValueError otherwise.

    Lowercase only, no path, no ``:443`` (browsers omit it from Origin), never a loopback or unspecified address
    (those are the local class), never IPv6 (the LAN entry point is IPv4, ADR-019).
    """
    origins: list[str] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        match = _PUBLIC_ORIGIN.fullmatch(item)
        if match is None or ".." in item:
            raise ValueError(f"{PUBLIC_ORIGINS_ENV}: not https://<lowercase host>[:port]: {item!r}")
        host, port = match["host"], match["port"]
        if port is not None and (port.startswith("0") or not 1 <= int(port) <= 65535 or port == "443"):
            raise ValueError(f"{PUBLIC_ORIGINS_ENV}: bad port (omit :443): {item!r}")
        if host == "localhost" or host.endswith(".localhost"):
            raise ValueError(f"{PUBLIC_ORIGINS_ENV}: localhost is the local class: {item!r}")
        try:
            address = ipaddress.IPv4Address(host)
        except ValueError:
            if host.replace(".", "").isdigit():
                raise ValueError(f"{PUBLIC_ORIGINS_ENV}: not an IPv4 address: {item!r}") from None
        else:
            if address.is_loopback or address.is_unspecified or address.is_multicast:
                raise ValueError(f"{PUBLIC_ORIGINS_ENV}: not a LAN address: {item!r}")
        if item not in origins:
            origins.append(item)
    return tuple(origins)


def parse_proxy_peer(raw: str) -> str | None:
    """``VLM_PROXY_PEER``: nginx's fixed IPv4 on the compose network, or None when unset; ValueError otherwise."""
    raw = raw.strip()
    return str(ipaddress.IPv4Address(raw)) if raw else None


# ------------------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class SecurityConfig:
    """What the middleware checks against; one instance per process (new secret on every restart)."""

    port: int
    dev: bool = False
    public_origins: tuple[str, ...] = ()
    proxy_peer: str | None = None
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    ws_auth_timeout_s: float = WS_AUTH_TIMEOUT_S

    @property
    def local_hosts(self) -> frozenset[str]:
        ports = [self.port, DEV_PORT] if self.dev else [self.port]
        return frozenset(f"{h}:{p}" for p in ports for h in ("127.0.0.1", "localhost"))

    @property
    def lan_hosts(self) -> dict[str, str]:
        """Host header -> its origin, for the LAN class."""
        return {origin.removeprefix("https://"): origin for origin in self.public_origins}

    def classify(self, host: str | None) -> HostClass | None:
        if host is None:
            return None
        if host in self.local_hosts:
            return "local"
        if host in self.lan_hosts:
            return "lan"
        return None

    def origin_of(self, host: str, cls: HostClass) -> str:
        return f"http://{host}" if cls == "local" else self.lan_hosts[host]

    def csrf_token(self, sid: str) -> str:
        return hmac.new(self.secret, b"csrf:" + sid.encode(), hashlib.sha256).hexdigest()

    def csrf_ok(self, sid: str, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value.encode(), self.csrf_token(sid).encode())


def cookie_name(cls: HostClass) -> str:
    return LAN_SESSION_COOKIE if cls == "lan" else SESSION_COOKIE


def _set_cookie(cls: HostClass, sid: str) -> bytes:
    secure = "; Secure" if cls == "lan" else ""
    return f"{cookie_name(cls)}={sid}; Max-Age={SESSION_ABSOLUTE_S}; HttpOnly; SameSite=Strict; Path=/{secure}".encode()


def _clear_cookie(cls: HostClass) -> bytes:
    secure = "; Secure" if cls == "lan" else ""
    return f"{cookie_name(cls)}=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/{secure}".encode()


# ----------------------------------------------------------------------------------------------------- sessions
@dataclass
class _Session:
    cls: HostClass
    created: float
    seen: float


class SessionStore:
    """In-memory sessions, keyed by sha256(id) so a lookup's timing says nothing about a live id."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self.max_sessions = max_sessions
        self._sessions: dict[str, _Session] = {}  # insertion order = creation order

    @staticmethod
    def _key(sid: str) -> str:
        return hashlib.sha256(sid.encode()).hexdigest()

    def _expired(self, s: _Session, now: float) -> bool:
        return now - s.seen > SESSION_IDLE_S or now - s.created > SESSION_ABSOLUTE_S

    def create(self, cls: HostClass) -> str:
        now = _now()
        for key in [k for k, s in self._sessions.items() if self._expired(s, now)]:
            del self._sessions[key]
        while len(self._sessions) >= self.max_sessions:
            del self._sessions[next(iter(self._sessions))]  # the oldest goes
        sid = secrets.token_urlsafe(32)
        self._sessions[self._key(sid)] = _Session(cls, now, now)
        return sid

    def check(self, sid: str | None, cls: HostClass) -> bool:
        """The session is live and was created for this Host class; refreshes its idle time."""
        if not sid or len(sid) > 128:
            return False
        key = self._key(sid)
        session = self._sessions.get(key)
        if session is None or session.cls != cls:
            return False
        now = _now()
        if self._expired(session, now):
            del self._sessions[key]
            return False
        session.seen = now
        return True

    def alive(self, sid: str, cls: HostClass) -> bool:
        """Like ``check`` without counting as use (an open WebSocket does not keep a session alive)."""
        session = self._sessions.get(self._key(sid))
        return session is not None and session.cls == cls and not self._expired(session, _now())

    def drop(self, sid: str) -> None:
        self._sessions.pop(self._key(sid), None)

    def clear(self) -> None:
        self._sessions.clear()

    def __len__(self) -> int:
        return len(self._sessions)


class LoginLimiter:
    """Failed LAN logins over the last 10 min: a warning at 20, LAN login locked 15 min at 200 (ADR-019)."""

    def __init__(self) -> None:
        self.failures: collections.deque[float] = collections.deque()
        self.locked_until: float | None = None
        self._warned = False

    def _prune(self, now: float) -> None:
        while self.failures and now - self.failures[0] > LOGIN_WINDOW_S:
            self.failures.popleft()
        if not self.failures:
            self._warned = False
        if self.locked_until is not None and now >= self.locked_until:
            self.locked_until = None

    def locked(self) -> float | None:
        self._prune(_now())
        return self.locked_until

    def fail(self) -> None:
        now = _now()
        self._prune(now)
        self.failures.append(now)
        if len(self.failures) >= LOGIN_WARN_FAILURES and not self._warned:
            self._warned = True
            log.warning("%d failed LAN logins in 10 min: someone may be guessing the token", len(self.failures))
        if len(self.failures) >= LOGIN_LOCK_FAILURES:
            self.locked_until = now + LOGIN_LOCK_S
            self.failures.clear()
            log.warning("LAN login locked for %d min after %d failures", LOGIN_LOCK_S // 60, LOGIN_LOCK_FAILURES)

    def reset(self) -> None:
        self.failures.clear()
        self.locked_until = None
        self._warned = False

    def status(self) -> dict[str, object]:
        return {"locked_until": self.locked(), "recent_failures": len(self.failures)}


class _Socket:
    """One open WebSocket under the middleware: closed with 4401 and its handler cancelled when its session ends."""

    def __init__(self, send: Send) -> None:
        self.send = send
        self.task: asyncio.Future[None] | None = None
        self.accepted = self.closed = self.revoked = False

    async def tracked_send(self, message: Message) -> None:
        if self.revoked:
            raise asyncio.CancelledError  # the handler is being cancelled: it sends nothing more
        kind = message.get("type")
        self.accepted |= kind == "websocket.accept"
        self.closed |= kind == "websocket.close"
        await self.send(message)

    async def revoke(self) -> None:
        if self.revoked:
            return
        self.revoked = True
        if self.accepted and not self.closed:
            self.closed = True
            with contextlib.suppress(Exception):  # the client may be gone already
                await self.send({"type": "websocket.close", "code": WS_CLOSE_UNAUTHORIZED})
        if self.task is not None:
            self.task.cancel()


class Guard:
    """The mutable side: the current token, the sessions, the LAN login counter."""

    def __init__(self, config: SecurityConfig, token: str, announce: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.token = token
        self.sessions = SessionStore()
        self.limiter = LoginLimiter()
        self.announce = announce
        self.sockets: dict[str, set[_Socket]] = {}  # sha256(session id) -> its open WebSockets
        self.ws_check_s = WS_SESSION_CHECK_S

    def watch(self, sid: str, sock: _Socket) -> None:
        self.sockets.setdefault(SessionStore._key(sid), set()).add(sock)

    def unwatch(self, sid: str, sock: _Socket) -> None:
        key = SessionStore._key(sid)
        socks = self.sockets.get(key, set())
        socks.discard(sock)
        if not socks:
            self.sockets.pop(key, None)

    async def close_sockets(self, sid: str | None) -> None:
        """Close (4401) the open WebSockets of one session, or of all sessions: signed out means no more events."""
        if sid is None:
            socks = [sock for group in self.sockets.values() for sock in group]
        else:
            socks = list(self.sockets.get(SessionStore._key(sid), ()))
        for sock in socks:
            await sock.revoke()

    def token_ok(self, given: object) -> bool:
        return isinstance(given, str) and hmac.compare_digest(given.encode(), self.token.encode())

    def rotate(self) -> None:
        """Sign out everywhere: every session goes, a new token (printed like at startup), the counter restarts."""
        self.sessions.clear()
        self.limiter.reset()
        self.token = secrets.token_urlsafe(32)
        if self.announce is not None:
            self.announce(self.token)

    def links(self, token: str | None = None) -> list[str]:
        token = token or self.token
        local = [f"http://127.0.0.1:{self.config.port}/#t={token}"]
        return local + [f"{origin}/#t={token}" for origin in self.config.public_origins]


# ---------------------------------------------------------------------------------------------------- the login page
_LOGIN_SCRIPT = """(function () {
  var form = document.getElementById("f"), input = document.getElementById("t"), out = document.getElementById("e");
  var m = /^#t=([A-Za-z0-9_-]{16,128})$/.exec(location.hash);
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
  function login(token) {
    out.textContent = "";
    fetch("/api/session", {method: "POST", credentials: "same-origin",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({token: token})})
      .then(function (r) {
        if (r.ok) { location.replace(location.pathname + location.search); return; }
        out.textContent = r.status === 429
          ? "Đăng nhập từ mạng LAN đang tạm khoá. / LAN sign-in is locked for now."
          : "Mã không đúng hoặc đã đổi. / Wrong or old code.";
      })
      .catch(function () { out.textContent = "Không kết nối được. / Cannot reach the app."; });
  }
  if (m) login(m[1]);
  form.addEventListener("submit", function (ev) { ev.preventDefault(); login(input.value.trim()); });
})();"""
_LOGIN_SCRIPT_HASH = base64.b64encode(hashlib.sha256(_LOGIN_SCRIPT.encode()).digest()).decode()
LOGIN_PAGE = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NTC Training - sign in</title>
<style>body{{font:16px system-ui,sans-serif;max-width:28rem;margin:4rem auto;padding:0 16px;color:#222;background:#fff}}
input,button{{font:inherit;padding:.5rem;margin:.25rem 0;width:100%;box-sizing:border-box}}#e{{color:#b00020}}
@media (prefers-color-scheme: dark){{body{{color:#eee;background:#16171a}}#e{{color:#ff8a80}}}}</style></head>
<body><h1>NTC Training</h1>
<p>Mở đường link in ra khi khởi động app, hoặc dán mã vào đây.<br>Open the link printed when the app started, or
paste its code here.</p>
<form id="f"><input id="t" type="password" autocomplete="off" aria-label="code"><button>Đăng nhập / Sign in</button>
</form><p id="e" role="alert"></p>
<script>{_LOGIN_SCRIPT}</script></body></html>
""".encode()
LOGIN_CSP = (
    f"default-src 'none'; script-src 'sha256-{_LOGIN_SCRIPT_HASH}'; style-src 'unsafe-inline'; connect-src 'self'; "
    "form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
).encode()


# ------------------------------------------------------------------------------------------------------ helpers
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


def _peer(scope: Scope) -> str | None:
    client = scope.get("client")
    return str(client[0]) if client else None


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


async def _login_page(send: Send) -> None:
    headers = [(b"content-type", b"text/html; charset=utf-8"), (b"content-length", str(len(LOGIN_PAGE)).encode()),
               (b"content-security-policy", LOGIN_CSP), *SECURITY_HEADERS]  # fmt: skip
    await send({"type": "http.response.start", "status": 401, "headers": headers})
    await send({"type": "http.response.body", "body": LOGIN_PAGE})


async def _read_body(receive: Receive, limit: int) -> bytes | None:
    body = b""
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            return None
        chunk: bytes = message.get("body", b"")
        body += chunk
        if len(body) > limit:
            return None
        if not message.get("more_body", False):
            return body


# --------------------------------------------------------------------------------------------------- middleware
class SecurityMiddleware:
    """Pure ASGI middleware (no framework dependency) enforcing D13 / ADR-019 for HTTP and WebSocket scopes."""

    def __init__(self, app: ASGIApp, guard: Guard) -> None:
        self.app = app
        self.guard = guard
        self.config = guard.config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            await self._http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
        else:  # lifespan
            await self.app(scope, receive, send)

    def _host_class(self, scope: Scope, headers: dict[str, str]) -> HostClass | None:
        """The request's Host class, or None (403): unknown Host, duplicated headers, or the wrong peer for it."""
        if _duplicated(scope, ("host", "origin", "cookie", CSRF_HEADER)):
            return None
        cls = self.config.classify(headers.get("host"))
        proxy = self.config.proxy_peer
        if proxy is not None and cls is not None and (_peer(scope) == proxy) != (cls == "lan"):
            return None  # a local Host through nginx (misconfigured proxy), or a LAN Host around it
        return cls

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

    async def _bootstrap(self, send: Send, method: str, given: list[str]) -> None:
        """``GET /?token=<startup token>`` (local Host only) -> session cookie + redirect: the token leaves the URL."""
        if method != "GET" or len(given) != 1 or not self.guard.token_ok(given[0]):
            await _respond(send, 401, {"code": "invalid_token"})
            return
        sid = self.guard.sessions.create("local")
        headers = [(b"location", b"./"), (b"set-cookie", _set_cookie("local", sid)), (b"content-length", b"0")]
        await send({"type": "http.response.start", "status": 303, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    async def _login(self, receive: Receive, send: Send, headers: dict[str, str], host: str, cls: HostClass) -> None:
        """``POST /api/session {"token": ...}``: exact Origin, JSON, constant-time compare, LAN failures counted."""
        guard = self.guard
        if headers.get("origin") != self.config.origin_of(host, cls):
            await _respond(send, 403, {"code": "forbidden_origin"})
            return
        if not headers.get("content-type", "").startswith("application/json"):
            await _respond(send, 415, {"code": "json_required"})
            return
        if cls == "lan" and (until := guard.limiter.locked()) is not None:
            await _respond(send, 429, {"code": "login_locked", "locked_until": until})
            return
        raw = await _read_body(receive, LOGIN_MAX_BYTES)
        try:
            given = json.loads(raw)["token"] if raw is not None else None
        except (ValueError, TypeError, KeyError):
            given = None
        if not guard.token_ok(given):
            if cls == "lan":
                guard.limiter.fail()
            await _respond(send, 401, {"code": "invalid_token"})
            return
        sid = guard.sessions.create(cls)
        log.info("signed in (%s), %d sessions", cls, len(guard.sessions))
        await _respond(send, 200, {"csrf_token": self.config.csrf_token(sid)}, [(b"set-cookie", _set_cookie(cls, sid))])

    async def _session_api(self, send: Send, method: str, path: str, sid: str, cls: HostClass) -> bool:
        """The session endpoints for a signed-in browser; False when ``path`` is not one of them."""
        guard = self.guard
        if path == SESSION_PATH and method == "GET":
            body: dict[str, object] = {"csrf_token": self.config.csrf_token(sid), "origin_class": cls}
            if cls == "local":  # only the owner's machine sees (and can lift) a LAN lock
                body["lan_login"] = guard.limiter.status()
            await _respond(send, 200, body)
        elif path == SESSION_PATH and method == "DELETE":
            guard.sessions.drop(sid)
            await guard.close_sockets(sid)
            await _respond(send, 200, {"ok": True}, [(b"set-cookie", _clear_cookie(cls))])
        elif path == REVOKE_ALL_PATH and method == "POST":
            log.warning("signed out everywhere (%s session): new token", cls)
            guard.rotate()
            await guard.close_sockets(None)
            await _respond(send, 200, {"ok": True}, [(b"set-cookie", _clear_cookie(cls))])
        elif path == UNLOCK_LAN_PATH and method == "POST":
            if cls != "local":
                await _respond(send, 403, {"code": "local_only"})
                return True
            guard.limiter.reset()
            log.warning("LAN login unlocked by local session")
            await _respond(send, 200, {"ok": True, "lan_login": guard.limiter.status()})
        else:
            return False
        return True

    def _write_problem(self, headers: dict[str, str], host: str, cls: HostClass, sid: str) -> str | None:
        """A state-changing request needs the Host's own Origin, then the session's CSRF token."""
        if headers.get("origin") != self.config.origin_of(host, cls):
            return "forbidden_origin"
        if not self.config.csrf_ok(sid, headers.get(CSRF_HEADER)):
            return "csrf_invalid"
        return None

    async def _http_checked(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = _headers(scope)
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        cls = self._host_class(scope, headers)
        if cls is None:
            await _respond(send, 403, {"code": "forbidden_host"})
            return
        host = headers["host"]
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        if path == "/" and "token" in query:
            if cls == "local":
                await self._bootstrap(send, method, query["token"])
            else:  # a LAN token never travels in a URL (proxy logs, history)
                await _respond(send, 401, {"code": "invalid_token"})
            return
        if path == SESSION_PATH and method == "POST":
            await self._login(receive, send, headers, host, cls)
            return
        sid = _cookie(headers.get("cookie"), cookie_name(cls))
        if sid is None or not self.guard.sessions.check(sid, cls):
            if method == "GET" and not (path == "/api" or path.startswith("/api/")):
                await _login_page(send)
            else:
                await _respond(send, 401, {"code": "unauthenticated", "hint": "open the URL printed in the terminal"})
            return
        if method in UNSAFE_METHODS and (problem := self._write_problem(headers, host, cls, sid)) is not None:
            await _respond(send, 403, {"code": problem})
            return
        if await self._session_api(send, method, path, sid, cls):
            return
        state = scope.setdefault("state", {})
        state["vlm_sid"], state["vlm_origin_class"] = sid, cls
        await self.app(scope, receive, send)

    # -------------------------------------------------------------------------------------------- websocket
    async def _websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = _headers(scope)
        cls = self._host_class(scope, headers)
        sid = _cookie(headers.get("cookie"), cookie_name(cls)) if cls is not None else None
        ok = (
            cls is not None
            and headers.get("origin") == self.config.origin_of(headers["host"], cls)
            and sid is not None
            and self.guard.sessions.check(sid, cls)
        )
        access_log.info("WS %s %s", _redacted_path(scope), "accepted-handshake" if ok else "rejected")
        if not ok or sid is None or cls is None:
            message = await receive()  # websocket.connect
            if message.get("type") == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})
            return
        state = scope.setdefault("state", {})
        state["vlm_security"], state["vlm_sid"], state["vlm_origin_class"] = self.guard, sid, cls
        await self._run_watched(scope, receive, send, sid, cls)

    async def _run_watched(self, scope: Scope, receive: Receive, send: Send, sid: str, cls: HostClass) -> None:
        """Run the handler as a task the guard can end: sign-out (one / everywhere) or session expiry closes it."""
        guard = self.guard
        sock = _Socket(send)
        sock.task = handler = asyncio.ensure_future(self.app(scope, receive, sock.tracked_send))

        async def expiry() -> None:
            while not handler.done():
                await asyncio.sleep(guard.ws_check_s)
                if not guard.sessions.alive(sid, cls):
                    await sock.revoke()
                    return

        guard.watch(sid, sock)
        watchdog = asyncio.ensure_future(expiry())
        try:
            await handler
        except asyncio.CancelledError:
            if not sock.revoked:
                raise  # the server cancelled us, not a sign-out
        finally:
            watchdog.cancel()
            guard.unwatch(sid, sock)


class _WebSocketLike(Protocol):
    scope: Any

    async def accept(self) -> None: ...

    async def receive_text(self) -> str: ...

    async def close(self, code: int = ...) -> None: ...


async def ws_authenticate(websocket: _WebSocketLike) -> None:
    """Accept, then require ``{"type": "auth", "csrf_token": ...}`` as the first message; else close 4401 and raise.

    Call at the start of every WebSocket endpoint, before sending anything. Raises ``PermissionError`` after closing.
    """
    state = websocket.scope.get("state", {})
    guard: Guard | None = state.get("vlm_security")
    sid, cls = state.get("vlm_sid"), state.get("vlm_origin_class")
    await websocket.accept()
    if guard is None or not isinstance(sid, str) or cls not in ("local", "lan"):  # middleware missing: fail closed
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        raise PermissionError("websocket security middleware not installed")
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=guard.config.ws_auth_timeout_s)
        if len(raw.encode()) > WS_AUTH_MAX_BYTES:
            raise ValueError("auth message too large")
        msg = json.loads(raw)
        if not (
            isinstance(msg, dict)
            and msg.get("type") == "auth"
            and guard.config.csrf_ok(sid, str(msg.get("csrf_token", "")))
            and guard.sessions.check(sid, cls)  # signed out while connecting
        ):
            raise ValueError("bad auth message")
    except (TimeoutError, ValueError, TypeError, KeyError) as exc:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED)
        raise PermissionError("websocket authentication failed") from exc


def _print_links(guard: Guard) -> Callable[[str], None]:
    def announce(token: str) -> None:
        for url in guard.links(token):
            print(f"vlm app: open {url}", file=sys.stderr, flush=True)  # shown, never logged (D13)

    return announce


def install_security(
    app: _App,
    *,
    port: int,
    dev: bool = False,
    public_origins: Sequence[str] = (),
    proxy_peer: str | None = None,
    token: str | None = None,
    secret: bytes | None = None,
    ws_auth_timeout_s: float = WS_AUTH_TIMEOUT_S,
) -> str:
    """Install the middleware on a FastAPI/Starlette app and return the startup token to print once.

    ``public_origins``/``proxy_peer`` come from ``VLM_PUBLIC_ORIGINS``/``VLM_PROXY_PEER`` (validated by the caller
    with ``parse_public_origins``/``parse_proxy_peer``). ``token``/``secret``/``ws_auth_timeout_s`` are for tests
    only; production generates a fresh random token and secret per process (a restart ends every session).
    """
    config = SecurityConfig(
        port=port,
        dev=dev,
        public_origins=parse_public_origins(",".join(public_origins)),
        proxy_peer=parse_proxy_peer(proxy_peer or ""),
        secret=secret or secrets.token_bytes(32),
        ws_auth_timeout_s=ws_auth_timeout_s,
    )
    guard = Guard(config, token or secrets.token_urlsafe(32))
    guard.announce = _print_links(guard)
    app.add_middleware(SecurityMiddleware, guard=guard)
    app.state.vlm_security = guard
    return guard.token


__all__ = [
    "ACCESS_LOGGER",
    "LAN_SESSION_COOKIE",
    "PROXY_PEER_ENV",
    "PUBLIC_ORIGINS_ENV",
    "SESSION_COOKIE",
    "WS_CLOSE_UNAUTHORIZED",
    "Guard",
    "SecurityConfig",
    "SecurityMiddleware",
    "install_security",
    "parse_proxy_peer",
    "parse_public_origins",
    "ws_authenticate",
]
