"""ADR-019 (D-8): LAN access through nginx. Host decides the class, sessions are bound to it, login by POST.

Security review 2026-10-13: never read X-Forwarded-*; ``__Host-`` cookie on the LAN; ``?token=`` local only;
LAN failures counted (warn 20, lock at 200), local login never locked and can lift the lock; sign out everywhere.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app import create_app, security
from vlm_pipeline.app.security import LAN_SESSION_COOKIE, SESSION_COOKIE, parse_public_origins

TOKEN = "t" * 43
PORT = 4003
LOCAL = f"127.0.0.1:{PORT}"
LAN = "192.168.2.178:4000"
LAN_ORIGIN = f"https://{LAN}"
NGINX = "172.30.0.10"


def _client(*, peer: str = "testclient", **kw: Any) -> TestClient:
    app = create_app(token=TOKEN, port=PORT, public_origins=[LAN_ORIGIN], **kw)
    return TestClient(app, base_url=f"http://{LOCAL}", client=(peer, 50000))


def _login(c: TestClient, host: str = LAN, origin: str = LAN_ORIGIN, token: str = TOKEN) -> Any:
    return c.post("/api/session", json={"token": token}, headers={"Host": host, "Origin": origin})


def _signed_in(c: TestClient) -> dict[str, str]:
    r = _login(c)
    assert r.status_code == 200, r.text
    c.cookies.clear()
    return {"Host": LAN, "Cookie": f"{LAN_SESSION_COOKIE}={r.cookies[LAN_SESSION_COOKIE]}",
            "Origin": LAN_ORIGIN, "X-VLM-CSRF": r.json()["csrf_token"]}  # fmt: skip


def _local(c: TestClient) -> dict[str, str]:
    r = _login(c, LOCAL, f"http://{LOCAL}")
    assert r.status_code == 200, r.text
    c.cookies.clear()
    return {"Host": LOCAL, "Cookie": f"{SESSION_COOKIE}={r.cookies[SESSION_COOKIE]}",
            "Origin": f"http://{LOCAL}", "X-VLM-CSRF": r.json()["csrf_token"]}  # fmt: skip


def test_a_lan_login_sets_a_host_prefixed_secure_cookie() -> None:
    c = _client()
    r = _login(c)
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{LAN_SESSION_COOKIE}=")
    assert all(flag in cookie for flag in ("Secure", "HttpOnly", "SameSite=Strict", "Path=/"))
    assert "Domain" not in cookie
    assert TOKEN not in cookie  # a session id, never the token
    headers = _signed_in(c)
    assert c.get("/api/system", headers=headers).status_code == 200
    assert c.get("/api/session", headers=headers).json()["origin_class"] == "lan"


def test_forwarded_headers_change_nothing() -> None:
    c = _client()
    spoof = {"X-Forwarded-Host": LAN, "X-Forwarded-Proto": "https", "X-Forwarded-For": "192.168.2.50"}
    r = c.post("/api/session", json={"token": TOKEN}, headers={"Host": "evil.example", "Origin": LAN_ORIGIN, **spoof})
    assert r.status_code == 403
    local = c.get(f"/?token={TOKEN}", headers={"Host": LOCAL, **spoof}, follow_redirects=False)
    assert local.headers["set-cookie"].startswith(f"{SESSION_COOKIE}=")  # still the local class


@pytest.mark.parametrize(
    ("origin", "status"),
    [("http://192.168.2.178:4000", 403), ("https://evil.example", 403), (None, 403)],
)
def test_login_needs_the_hosts_own_origin(origin: str | None, status: int) -> None:
    headers = {"Host": LAN} | ({"Origin": origin} if origin else {})
    assert _client().post("/api/session", json={"token": TOKEN}, headers=headers).status_code == status


def test_login_needs_json() -> None:
    form = {"Host": LAN, "Origin": LAN_ORIGIN, "Content-Type": "application/x-www-form-urlencoded"}
    r = _client().post("/api/session", content=f"token={TOKEN}", headers=form)
    assert r.status_code == 415


def test_the_url_token_is_refused_on_the_lan() -> None:
    c = _client()
    r = c.get(f"/?token={TOKEN}", headers={"Host": LAN}, follow_redirects=False)
    assert r.status_code == 401
    assert "set-cookie" not in r.headers
    local = c.get(f"/?token={TOKEN}", headers={"Host": LOCAL}, follow_redirects=False)
    assert (local.status_code, local.headers["location"]) == (303, "./")


def test_a_session_works_only_for_the_class_it_was_made_for() -> None:
    c = _client()
    lan_sid = _signed_in(c)["Cookie"].split("=", 1)[1]
    local_sid = _local(c)["Cookie"].split("=", 1)[1]
    assert c.get("/api/system", headers={"Host": LOCAL, "Cookie": f"{SESSION_COOKIE}={lan_sid}"}).status_code == 401
    assert c.get("/api/system", headers={"Host": LAN, "Cookie": f"{LAN_SESSION_COOKIE}={local_sid}"}).status_code == 401
    assert c.get("/api/system", headers={"Host": LAN, "Cookie": f"{SESSION_COOKIE}={lan_sid}"}).status_code == 401


def test_writes_need_the_lan_origin_and_csrf() -> None:
    c = _client()
    headers = _signed_in(c)
    assert c.post("/api/graphs/validate", json={}, headers=headers).status_code == 200
    assert c.post("/api/graphs/validate", json={}, headers={**headers, "Origin": f"http://{LAN}"}).status_code == 403
    assert c.post("/api/graphs/validate", json={}, headers={**headers, "X-VLM-CSRF": "x"}).status_code == 403


def test_the_websocket_works_over_the_lan() -> None:
    c = _client(ws_auth_timeout_s=0.5)
    headers = _signed_in(c)
    csrf = headers.pop("X-VLM-CSRF")
    with c.websocket_connect("/api/runs/x/events", headers=headers) as ws:
        ws.send_text('{"type": "auth", "csrf_token": "' + csrf + '"}')
        assert isinstance(ws.receive_json(), dict)


def test_an_unauthenticated_browser_gets_the_login_page() -> None:
    r = _client().get("/", headers={"Host": LAN})
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    script_src = csp.split("script-src")[1].split(";")[0]
    assert script_src.strip().startswith("'sha256-")
    assert "unsafe-inline" not in script_src
    assert "location.hash" in r.text
    assert TOKEN not in r.text
    api = _client().get("/api/system", headers={"Host": LAN})
    assert (api.status_code, api.json()["code"]) == (401, "unauthenticated")


def test_failed_lan_logins_warn_then_lock_but_local_never_locks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = [1_000_000.0]
    monkeypatch.setattr(security, "_now", lambda: clock[0])
    caplog.set_level(logging.WARNING, logger="vlm_pipeline.app.security")
    c = _client()
    for _ in range(security.LOGIN_WARN_FAILURES):
        assert _login(c, token="w" * 43).status_code == 401
    assert "may be guessing" in caplog.text
    for _ in range(security.LOGIN_LOCK_FAILURES - security.LOGIN_WARN_FAILURES):
        _login(c, token="w" * 43)
    locked = _login(c)
    assert (locked.status_code, locked.json()["code"]) == (429, "login_locked")  # even with the right token
    local = _local(c)  # the owner always gets in
    status = c.get("/api/session", headers=local).json()["lan_login"]
    assert status["locked_until"] == clock[0] + security.LOGIN_LOCK_S
    clock[0] += security.LOGIN_LOCK_S + 1
    assert _login(c).status_code == 200  # the lock ends by itself


def test_only_a_local_session_lifts_the_lan_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    lan = _signed_in(c)
    guard = c.app.state.vlm_security  # type: ignore[attr-defined]
    for _ in range(security.LOGIN_LOCK_FAILURES):
        guard.limiter.fail()
    assert _login(c).status_code == 429
    refused = c.post("/api/session/unlock-lan", headers=lan)
    assert (refused.status_code, refused.json()["code"]) == (403, "local_only")
    ok = c.post("/api/session/unlock-lan", headers=_local(c))
    assert ok.status_code == 200
    assert ok.json()["lan_login"] == {"locked_until": None, "recent_failures": 0}
    assert _login(c).status_code == 200
    assert "lan_login" not in c.get("/api/session", headers=lan).json()  # a LAN session never sees the lock


def test_sign_out_and_sign_out_everywhere() -> None:
    c = _client()
    guard = c.app.state.vlm_security  # type: ignore[attr-defined]
    printed: list[str] = []
    guard.announce = printed.append
    one, two = _signed_in(c), _signed_in(c)
    out = c.delete("/api/session", headers=one)
    assert out.status_code == 200
    assert "Max-Age=0" in out.headers["set-cookie"]
    assert c.get("/api/system", headers=one).status_code == 401
    assert c.get("/api/system", headers=two).status_code == 200
    local = _local(c)
    assert c.post("/api/session/revoke-all", headers=two).status_code == 200
    assert c.get("/api/system", headers=local).status_code == 401  # every session, both classes
    assert _login(c).status_code == 401  # the old token is gone
    assert len(printed) == 1
    assert printed[0] != TOKEN
    assert _login(c, token=printed[0]).status_code == 200


def test_sessions_expire_and_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1_000_000.0]
    monkeypatch.setattr(security, "_now", lambda: clock[0])
    c = _client()
    first = _signed_in(c)
    clock[0] += security.SESSION_IDLE_S - 10
    assert c.get("/api/system", headers=first).status_code == 200  # use refreshes the idle time
    clock[0] += security.SESSION_IDLE_S - 10
    assert c.get("/api/system", headers=first).status_code == 200
    clock[0] += security.SESSION_IDLE_S + 1
    assert c.get("/api/system", headers=first).status_code == 401  # idle
    long = _signed_in(c)
    for _ in range(15):
        clock[0] += security.SESSION_IDLE_S - 10
        c.get("/api/system", headers=long)
    assert c.get("/api/system", headers=long).status_code == 401  # 7 days absolute
    oldest = _signed_in(c)
    for _ in range(security.MAX_SESSIONS):
        _signed_in(c)
    assert c.get("/api/system", headers=oldest).status_code == 401


def test_with_a_proxy_peer_each_class_comes_from_its_own_side() -> None:
    via_nginx = _client(peer=NGINX, proxy_peer=NGINX)
    assert _login(via_nginx).status_code == 200
    assert _login(via_nginx, LOCAL, f"http://{LOCAL}").status_code == 403  # local Host through nginx: misconfigured
    direct = _client(peer="172.30.0.1", proxy_peer=NGINX)
    assert _login(direct, LOCAL, f"http://{LOCAL}").status_code == 200
    assert _login(direct).status_code == 403  # LAN Host around nginx


@pytest.mark.parametrize(
    "raw",
    ["http://192.168.2.178:4000", "https://192.168.2.178:4000/", "https://127.0.0.1:4000", "https://localhost:4000",
     "https://0.0.0.0:4000", "https://HOST:4000", "https://192.168.2.178:443", "https://192.168.2.178:99999",
     "https://a..b:4000", "https://[::1]:4000", "https://999.1.1.1:4000"],
)  # fmt: skip
def test_bad_public_origins_stop_the_app(raw: str) -> None:
    with pytest.raises(ValueError, match="VLM_PUBLIC_ORIGINS"):
        parse_public_origins(raw)


def test_good_public_origins() -> None:
    raw = "https://192.168.2.178:4000, https://spark.lan:4000,https://spark.lan:4000,https://spark.lan"
    assert parse_public_origins(raw) == ("https://192.168.2.178:4000", "https://spark.lan:4000", "https://spark.lan")
    assert parse_public_origins("") == ()


def _open_ws(stack: contextlib.ExitStack, c: TestClient, headers: dict[str, str]) -> Any:
    headers = dict(headers)
    csrf = headers.pop("X-VLM-CSRF")
    ws = stack.enter_context(c.websocket_connect("/api/runs/x/events", headers=headers))
    ws.send_text('{"type": "auth", "csrf_token": "' + csrf + '"}')
    return ws


def _close_code(ws: Any) -> int:
    from starlette.websockets import WebSocketDisconnect

    def drain() -> None:
        while True:
            ws.receive_json()

    with pytest.raises(WebSocketDisconnect) as closed:
        drain()
    return int(closed.value.code)


def _until(check: Any) -> None:
    for _ in range(200):
        if check():
            return
        threading.Event().wait(0.02)
    raise AssertionError("condition never became true")


def _blocking_bus(c: TestClient) -> None:
    """Every run id is known and never ends: the socket stays open until the guard closes it."""
    import asyncio

    bus = c.app.state.bus  # type: ignore[attr-defined]
    bus.known = lambda _id: True

    async def follow(_id: str) -> Any:
        await asyncio.Event().wait()
        yield {}  # pragma: no cover

    bus.follow = follow


def _open_sockets(guard: Any) -> int:
    return sum(len(group) for group in guard.sockets.values())


def test_sign_out_everywhere_closes_open_websockets() -> None:
    with _client(ws_auth_timeout_s=2) as c, contextlib.ExitStack() as stack:
        _blocking_bus(c)
        guard = c.app.state.vlm_security  # type: ignore[attr-defined]
        guard.announce = lambda _t: None
        lan, local = _signed_in(c), _local(c)
        ws_lan, ws_local = _open_ws(stack, c, lan), _open_ws(stack, c, local)
        _until(lambda: _open_sockets(guard) == 2)
        assert c.post("/api/session/revoke-all", headers=local).status_code == 200
        assert _close_code(ws_lan) == security.WS_CLOSE_UNAUTHORIZED
        assert _close_code(ws_local) == security.WS_CLOSE_UNAUTHORIZED
        _until(lambda: guard.sockets == {})


def test_sign_out_closes_only_that_sessions_websockets() -> None:
    with _client(ws_auth_timeout_s=2) as c, contextlib.ExitStack() as stack:
        _blocking_bus(c)
        guard = c.app.state.vlm_security  # type: ignore[attr-defined]
        one, two = _signed_in(c), _signed_in(c)
        ws_one = _open_ws(stack, c, one)
        _open_ws(stack, c, two)
        _until(lambda: _open_sockets(guard) == 2)
        assert c.delete("/api/session", headers=one).status_code == 200
        assert _close_code(ws_one) == security.WS_CLOSE_UNAUTHORIZED
        _until(lambda: _open_sockets(guard) == 1)  # the other session's socket stays open


def test_an_expired_session_closes_its_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1_000_000.0]
    monkeypatch.setattr(security, "_now", lambda: clock[0])
    with _client(ws_auth_timeout_s=2) as c, contextlib.ExitStack() as stack:
        _blocking_bus(c)
        guard = c.app.state.vlm_security  # type: ignore[attr-defined]
        guard.ws_check_s = 0.05
        ws = _open_ws(stack, c, _signed_in(c))
        _until(lambda: _open_sockets(guard) == 1)
        threading.Event().wait(0.2)
        assert _open_sockets(guard) == 1  # still open: the session is alive
        clock[0] += security.SESSION_IDLE_S + 1
        assert _close_code(ws) == security.WS_CLOSE_UNAUTHORIZED


def test_the_lan_settings_the_server_starts_with(monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.app.server import lan_settings

    monkeypatch.delenv("VLM_IN_COMPOSE", raising=False)
    monkeypatch.delenv("VLM_PROXY_PEER", raising=False)
    monkeypatch.delenv("VLM_PUBLIC_ORIGINS", raising=False)
    assert lan_settings() == ((), None)
    monkeypatch.setenv("VLM_PUBLIC_ORIGINS", LAN_ORIGIN)
    assert lan_settings() == ((LAN_ORIGIN,), None)  # dev: no nginx peer to pin
    monkeypatch.setenv("VLM_IN_COMPOSE", "1")
    assert "VLM_PROXY_PEER" in str(lan_settings())  # compose: never a silent "no peer check"
    monkeypatch.setenv("VLM_PROXY_PEER", NGINX)
    assert lan_settings() == ((LAN_ORIGIN,), NGINX)
    monkeypatch.setenv("VLM_PROXY_PEER", "nginx")
    assert isinstance(lan_settings(), str)
    monkeypatch.setenv("VLM_PUBLIC_ORIGINS", "http://192.168.2.178:4000")
    assert isinstance(lan_settings(), str)
