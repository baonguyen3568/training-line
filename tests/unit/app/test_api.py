from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import NotAvailableError
from vlm_pipeline.nodes.spec import NodeSpec, PortSpec, Registry


class FakeServices:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = Registry(
            [
                NodeSpec(type_id="data.fake", version="1.0.0", category="data", icon="db", runner="host",
                         outputs={"dataset": PortSpec(type="Dataset")}),
                NodeSpec(type_id="train.fake", version="1.0.0", category="train", icon="bolt", runner="host",
                         inputs={"dataset": PortSpec(type="Dataset")}, outputs={"model": PortSpec(type="Checkpoint")}),
            ]
        )  # fmt: skip
        self.gate = threading.Event()

    def validate(self, workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
        if workflow.get("bad"):
            return [{"node_id": "n1", "port": "dataset", "param": None, "code": "CFG-003", "severity": "error",
                     "params": {"node": "n1", "port": "dataset"}}]  # fmt: skip
        return []

    def run_graph(self, workflow: Mapping[str, Any], graph_run_id: str, on_event: Any) -> None:
        for node in ("n1", "n2"):
            on_event({"type": "node_status", "node_id": node, "status": "running"})
            on_event({"type": "node_status", "node_id": node, "status": "done"})
        self.gate.wait(5)
        on_event({"type": "run_status", "status": "completed"})

    def cancel(self, graph_run_id: str) -> None:
        raise NotAvailableError("todo")

    def list_runs(self) -> list[dict[str, Any]]:
        from vlm_pipeline.artifacts.runs_index import list_runs

        return [m.model_dump(mode="json") for m in list_runs(self.root)]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return next((r for r in self.list_runs() if r["run_id"] == run_id), None)

    def metrics_page(self, run_id: str, after: str | None, limit: int, downsample_to: int | None) -> dict[str, Any]:
        from vlm_pipeline.artifacts.runs_index import read_metrics_page

        return read_metrics_page(self.root / "outputs" / "runs" / run_id, after, limit, downsample_to).model_dump()

    def system(self) -> dict[str, Any]:
        return {"mem_total_gb": 121.0, "mem_available_gb": 40.0, "disk_free_gb": 2000.0, "gpu_lock": {"held": False}}


@pytest.fixture
def svc(tmp_path: Path) -> FakeServices:
    from vlm_pipeline.artifacts.testing import seed_demo_runs

    seed_demo_runs(tmp_path / "outputs" / "runs")
    return FakeServices(tmp_path)


@pytest.fixture
def client(svc: FakeServices) -> TestClient:
    return TestClient(create_app(svc, secure=False))


def test_node_types_with_locale(client: TestClient) -> None:
    body = client.get("/api/node-types", params={"locale": "en"}).json()
    ids = [n["type_id"] for n in body["node_types"]]
    assert ids == ["data.fake", "train.fake"]
    node = body["node_types"][0]
    assert node["title_key"] == "node.data.fake.title"
    assert node["title"]  # resolved text or the key itself, never missing
    assert {p["id"] for p in body["port_types"]} >= {"Dataset", "Model", "Checkpoint"}
    assert client.get("/api/nodes").status_code == 200
    assert client.get("/api/node-types", params={"locale": "fr"}).status_code == 422


def test_validate_renders_text_but_keeps_code(client: TestClient) -> None:
    errors = client.post("/api/graphs/validate", json={"bad": True}).json()["errors"]
    assert errors[0]["code"] == "CFG-003"
    assert errors[0]["title_key"] == "error.CFG-003.title"
    assert errors[0]["title"]
    assert client.post("/api/graphs/validate", json={"nodes": []}).json() == {"errors": []}


def test_invalid_graph_is_not_started(client: TestClient) -> None:
    r = client.post("/api/runs", json={"bad": True})
    assert r.status_code == 422


def test_run_streams_events_over_ws_with_backlog(client: TestClient, svc: FakeServices) -> None:
    graph_run_id = client.post("/api/runs", json={"nodes": []}).json()["graph_run_id"]
    time.sleep(0.05)  # events published before the client connects must be replayed
    with client.websocket_connect(f"/api/runs/{graph_run_id}/events") as ws:
        first = [ws.receive_json() for _ in range(4)]
        svc.gate.set()
        last = ws.receive_json()
    assert [e["status"] for e in first] == ["running", "done", "running", "done"]
    assert all(e["graph_run_id"] == graph_run_id and "ts" in e for e in first)
    assert last == {**last, "type": "run_status", "status": "completed"}


def test_unknown_run_ws_closes_4404(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as ei, client.websocket_connect("/api/runs/nope/events") as ws:
        assert ws.receive_json() == {"type": "run_not_found", "graph_run_id": "nope"}
        ws.receive_json()
    assert ei.value.code == 4404


def test_runs_and_metrics_from_seeded_dirs(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert {r["status"] for r in runs} >= {"completed", "failed", "running"}
    run_id = next(r["run_id"] for r in runs if r["status"] == "completed")
    assert client.get(f"/api/runs/{run_id}").json()["run_id"] == run_id
    assert client.get("/api/runs/does-not-exist").status_code == 404
    page = client.get(f"/api/runs/{run_id}/metrics", params={"limit": 3}).json()
    assert len(page["points"]) <= 3
    assert set(page) == {"points", "next_cursor", "has_more", "downsampled"}
    assert client.get(f"/api/runs/{run_id}/metrics", params={"after": "x"}).status_code == 400


def test_not_available_is_501(client: TestClient) -> None:
    r = client.post("/api/runs/any/cancel")
    assert r.status_code == 501


def test_system(client: TestClient) -> None:
    assert client.get("/api/system").json()["gpu_lock"] == {"held": False}
