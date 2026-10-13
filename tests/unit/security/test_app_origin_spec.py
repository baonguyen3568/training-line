"""Spec tests for the local web app hardening (P003 D13): Host allowlist, Origin check, session token.

ASSUMED: ``vlm_pipeline.app.create_app(token: str, port: int, secret: bytes | None = None,
ws_auth_timeout_s: float = 5.0)`` returns a FastAPI app; tested with
``fastapi.testclient.TestClient`` (no network, no real port). Access log goes through the stdlib logger
``vlm_pipeline.access`` (app middleware), so it is checkable with ``caplog``. Rename together with bao-backend.
CSRF (D13, after the S-5 review): the CSRF token is NOT the session token; ``GET /api/session`` returns
``{"csrf_token": HMAC(secret, session_id)}``. WebSocket: after the handshake the first client message must carry the
CSRF token (``{"type": "auth", "csrf_token": ...}``); a wrong token, a first message > 4 KB, or silence past
``ws_auth_timeout_s`` closes with code 4401 (after accept). A handshake with a bad Origin or no session cookie is
refused with 1008 before accept. Security side: ``app/security.py`` (``install_security``, ``ws_authenticate``).
"""

from __future__ import annotations

from typing import Any

import pytest

TOKEN = "t" * 32
PORT = 7870
GOOD_HOST = f"127.0.0.1:{PORT}"


def _client(**kwargs: Any):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from vlm_pipeline.app import create_app

    return TestClient(create_app(token=TOKEN, port=PORT, **kwargs), base_url=f"http://{GOOD_HOST}")


def _base(client: Any) -> dict[str, str]:
    """A signed-in local session (ADR-019: the cookie is a random session id, never the token)."""
    r = client.get(f"/?token={TOKEN}", headers={"Host": GOOD_HOST}, follow_redirects=False)
    assert r.status_code == 303
    client.cookies.clear()  # headers below carry the cookie explicitly
    return {"Host": GOOD_HOST, "Cookie": f"vlm_session={r.cookies['vlm_session']}"}


def _auth(client: Any) -> dict[str, str]:
    base = _base(client)
    csrf = client.get("/api/session", headers=base).json()["csrf_token"]
    return {**base, "X-VLM-CSRF": csrf}


def _post_validate(client: Any, headers: dict[str, str]) -> Any:
    return client.post("/api/graphs/validate", json={}, headers={**headers, "Origin": f"http://{GOOD_HOST}"})


@pytest.mark.parametrize(
    "host",
    [
        "evil.example:7870",
        "192.168.1.5:7870",
        f"127.0.0.1:{PORT + 1}",
        f"127.0.0.1.attacker.com:{PORT}",  # exact match, never startswith
        f"localhost.evil:{PORT}",
        "127.0.0.1",  # port must match too
    ],
)
def test_wrong_host_header_is_403(host: str) -> None:
    """DNS rebinding defence."""
    c = _client()
    r = c.get("/api/system", headers={**_auth(c), "Host": host})
    assert r.status_code == 403


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}"])
def test_allowed_hosts(host: str) -> None:
    c = _client()
    r = c.get("/api/system", headers={**_auth(c), "Host": host})
    assert r.status_code == 200


def test_state_change_with_foreign_origin_is_403() -> None:
    c = _client()
    r = c.post("/api/graphs/validate", json={}, headers={**_auth(c), "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_state_change_without_csrf_header_is_403() -> None:
    c = _client()
    headers = {k: v for k, v in _auth(c).items() if k != "X-VLM-CSRF"}
    r = c.post("/api/graphs/validate", json={}, headers={**headers, "Origin": f"http://{GOOD_HOST}"})
    assert r.status_code == 403


def test_no_token_is_401() -> None:
    r = _client().get("/api/system", headers={"Host": GOOD_HOST})
    assert r.status_code == 401


def test_the_token_itself_is_not_a_session_cookie() -> None:
    r = _client().get("/api/system", headers={"Host": GOOD_HOST, "Cookie": f"vlm_session={TOKEN}"})
    assert r.status_code == 401


def test_no_cors_headers() -> None:
    r = _client().options(
        "/api/runs",
        headers={"Host": GOOD_HOST, "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


@pytest.mark.parametrize(
    ("signed_in", "origin"),
    [
        (False, f"http://{GOOD_HOST}"),  # no session cookie
        (True, "http://evil.example"),  # foreign Origin
        (True, f"https://{GOOD_HOST}"),  # not the Host's own scheme
    ],
)
def test_websocket_handshake_is_refused_with_1008(signed_in: bool, origin: str) -> None:
    from starlette.websockets import WebSocketDisconnect

    c = _client()
    headers = {**(_base(c) if signed_in else {"Host": GOOD_HOST}), "Origin": origin}
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        c.websocket_connect("/api/runs/x/events", headers=headers),
    ):
        pass
    assert refused.value.code == 1008


def test_token_query_param_not_in_access_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vlm_pipeline.access")
    _client().get(f"/?token={TOKEN}", headers={"Host": GOOD_HOST})
    assert caplog.records, "access log must exist (otherwise this test proves nothing)"
    assert TOKEN not in caplog.text


def test_spoofed_package_hint_not_echoed_by_api() -> None:
    """P003 D16: the hint must not appear anywhere in the API response, technical_detail included."""
    import json

    hint = "vlm/lora-trainer"
    wf = {
        "format": "vlm-workflow",
        "schema_version": 1,
        "name": "t",
        "nodes": [{"id": "x", "type": "acme/fake.train", "version": "1.0.0", "params": {}, "package": hint}],
        "edges": [],
    }
    c = _client()
    r = c.post("/api/graphs/validate", json=wf, headers={**_auth(c), "Origin": f"http://{GOOD_HOST}"})
    assert r.status_code == 200
    body = r.json()
    assert hint not in json.dumps(body)
    cfg004 = [i for i in body["errors"] if i["code"] == "CFG-004"]
    assert cfg004
    assert cfg004[0]["params"]["suggested_package"] is None


def test_session_token_is_not_accepted_as_csrf() -> None:
    c = _client()
    assert _post_validate(c, {**_base(c), "X-VLM-CSRF": TOKEN}).status_code == 403
    assert _post_validate(c, _auth(c)).status_code == 200  # sanity: the real CSRF token works


def test_csrf_from_another_app_instance_is_rejected() -> None:
    """A token minted with another secret (e.g. before a restart) must not be accepted."""
    old = _client(secret=b"a" * 32)
    new = _client(secret=b"b" * 32)
    stale = _auth(old)["X-VLM-CSRF"]
    assert _post_validate(new, {**_base(new), "X-VLM-CSRF": stale}).status_code == 403


def _ws_close_code(client: Any, first_message: str | None) -> int:
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect(
        "/api/runs/x/events", headers={**_base(client), "Origin": f"http://{GOOD_HOST}"}
    ) as ws:
        if first_message is not None:
            ws.send_text(first_message)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    return int(closed.value.code)


@pytest.mark.parametrize(
    "first_message",
    [
        '{"type": "auth", "csrf_token": "wrong"}',
        '{"type": "auth", "csrf_token": "' + TOKEN + '"}',  # session token is not the CSRF token
        '{"type": "auth", "csrf_token": "' + "x" * 5000 + '"}',  # > 4 KB
        None,  # silence past the auth timeout
    ],
)
def test_websocket_first_message_must_authenticate(first_message: str | None) -> None:
    assert _ws_close_code(_client(ws_auth_timeout_s=0.2), first_message) == 4401


def test_websocket_with_valid_csrf_stays_open() -> None:
    c = _client(ws_auth_timeout_s=0.2)
    auth = _auth(c)
    csrf = auth.pop("X-VLM-CSRF")
    with c.websocket_connect("/api/runs/x/events", headers={**auth, "Origin": f"http://{GOOD_HOST}"}) as ws:
        ws.send_text('{"type": "auth", "csrf_token": "' + csrf + '"}')
        message = ws.receive_json()  # first server message (e.g. snapshot / run_not_found), not a 4401 close
    assert isinstance(message, dict)
