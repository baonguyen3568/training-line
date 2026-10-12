"""Spec tests for the local web app hardening (P003 D13): Host allowlist, Origin check, session token.

ASSUMED: ``vlm_pipeline.app.create_app(token: str, port: int)`` returns a FastAPI app; tested with
``fastapi.testclient.TestClient`` (no network, no real port). Access log goes through the stdlib logger
``vlm_pipeline.access`` (app middleware), so it is checkable with ``caplog``. Rename together with bao-backend.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.app (P009/P003 D13) in P2"
)

TOKEN = "t" * 32
PORT = 7870
GOOD_HOST = f"127.0.0.1:{PORT}"


def _client():  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from vlm_pipeline.app import create_app

    return TestClient(create_app(token=TOKEN, port=PORT), base_url=f"http://{GOOD_HOST}")


def _auth() -> dict[str, str]:
    return {"Host": GOOD_HOST, "Cookie": f"vlm_session={TOKEN}", "X-VLM-CSRF": TOKEN}


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
    r = _client().get("/api/system", headers={**_auth(), "Host": host})
    assert r.status_code == 403


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}"])
def test_allowed_hosts(host: str) -> None:
    r = _client().get("/api/system", headers={**_auth(), "Host": host})
    assert r.status_code == 200


def test_state_change_with_foreign_origin_is_403() -> None:
    r = _client().post("/api/graphs/validate", json={}, headers={**_auth(), "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_state_change_without_csrf_header_is_403() -> None:
    headers = {k: v for k, v in _auth().items() if k != "X-VLM-CSRF"}
    r = _client().post("/api/graphs/validate", json={}, headers={**headers, "Origin": f"http://{GOOD_HOST}"})
    assert r.status_code == 403


def test_no_token_is_401() -> None:
    r = _client().get("/api/system", headers={"Host": GOOD_HOST})
    assert r.status_code == 401


def test_no_cors_headers() -> None:
    r = _client().options(
        "/api/runs",
        headers={"Host": GOOD_HOST, "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_websocket_without_cookie_is_rejected() -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), _client().websocket_connect(
        "/api/runs/x/events", headers={"Host": GOOD_HOST, "Origin": f"http://{GOOD_HOST}"}
    ):
        pass


def test_websocket_with_foreign_origin_is_rejected() -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), _client().websocket_connect(
        "/api/runs/x/events", headers={**_auth(), "Origin": "http://evil.example"}
    ):
        pass


def test_token_query_param_not_in_access_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="vlm_pipeline.access")
    _client().get(f"/?token={TOKEN}", headers={"Host": GOOD_HOST})
    assert caplog.records, "access log must exist (otherwise this test proves nothing)"
    assert TOKEN not in caplog.text
