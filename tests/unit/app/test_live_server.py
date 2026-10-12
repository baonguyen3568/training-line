"""Real uvicorn + real WebSocket library (TestClient bypasses both): bootstrap -> CSRF -> run -> events."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import websockets

from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.graph.testing import make_test_registry

TOKEN = "k" * 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _workflow() -> dict[str, Any]:
    def n(i: str, t: str, **p: Any) -> dict[str, Any]:
        return {"id": i, "type": t, "version": "1.0.0", "params": p}

    return {
        "format": "vlm-workflow",
        "schema_version": 1,
        "name": "live",
        "nodes": [n("d", "test.dataset"), n("m", "test.model"), n("t", "test.train", max_steps=10)],
        "edges": [{"from": ["d", "data"], "to": ["t", "data"]}, {"from": ["m", "model"], "to": ["t", "base_model"]}],
    }


@pytest.fixture
def server(tmp_path: Path) -> Any:
    port = _free_port()
    services = DefaultServices(registry=make_test_registry(), root=tmp_path, fake_delay_s=0.05)
    app = create_app(services, token=TOKEN, port=port)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    yield port
    srv.should_exit = True
    thread.join(5)


def test_run_events_reach_a_real_websocket_client(server: int) -> None:
    base = f"http://127.0.0.1:{server}"
    with httpx.Client(base_url=base, follow_redirects=False) as http:
        http.get(f"/?token={TOKEN}")  # bootstrap: token -> HttpOnly session cookie
        cookie = http.cookies.get("vlm_session")
        assert cookie
        csrf = http.get("/api/session").json()["csrf_token"]
        headers = {"X-VLM-CSRF": csrf, "Origin": base}
        r = http.post("/api/runs", json=_workflow(), headers=headers)
        assert r.status_code == 202, r.text
        gid = r.json()["graph_run_id"]

    async def collect() -> list[dict[str, Any]]:
        url = f"ws://127.0.0.1:{server}/api/runs/{gid}/events"
        events = []
        async with websockets.connect(url, origin=base, additional_headers={"Cookie": f"vlm_session={cookie}"}) as ws:  # type: ignore[arg-type]
            await ws.send(json.dumps({"type": "auth", "csrf_token": csrf}))
            async for raw in ws:
                events.append(json.loads(raw))
        return events

    events = asyncio.run(asyncio.wait_for(collect(), timeout=20))
    statuses = {(e.get("node_id"), e.get("status")) for e in events if e.get("type") == "node_status"}
    assert {("d", "completed"), ("m", "completed"), ("t", "completed")} <= statuses
    final = [e for e in events if e.get("type") == "run_status"][-1]
    assert final["status"] == "completed"
