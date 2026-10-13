"""Storage panel "Reset this step" routes (G-4): dry-run plan, typed confirmation, approval threshold, refusals."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlm_pipeline.app.storage import StorageApi, StorageSettings, register_storage_routes
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.common.run_claims import node_generation

NODE = "n1-0123456789abcdef"


def _attempt(root: Path, name: str, status: str = "failed", *, ckpt: bool = True) -> Path:
    run = root / "outputs" / "graph" / "runs" / name
    run.mkdir(parents=True)
    if ckpt:
        (run / "checkpoints" / "epoch_0_step_9").mkdir(parents=True)
        (run / "checkpoints" / "epoch_0_step_9" / "w").write_bytes(b"w" * 4096)
    manifest = {"run_id": name, "name": "train.lora_vlm Qwen", "status": status, "ended_at": time.time()}
    (run / "run.json").write_text(json.dumps(manifest))
    return run


def _client(root: Path, settings: StorageSettings | None = None) -> TestClient:
    (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(root / "outputs" / "graph" / "app.db")
    api = StorageApi(root, idx, settings or StorageSettings(node_cache_max_gb=0.0), lock_path=root / "no-lock")
    app = FastAPI()
    register_storage_routes(app, api)
    return TestClient(app)


def test_reset_needs_the_node_name_typed_when_checkpoints_go(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _attempt(tmp_path, NODE)
    _attempt(tmp_path, f"{NODE}-a2", "interrupted")
    plan = client.get("/api/storage/node-reset", params={"node": NODE, "locale": "vi"}).json()
    assert [a["run_dir"] for a in plan["attempts"]] == [NODE, f"{NODE}-a2"]
    title = plan["node_title"]
    assert title != NODE  # a person types the step's title, never a dir name (bao-design)
    assert (plan["consequence_code"], plan["confirm_text"], plan["next_generation"]) == ("PERMANENT_LOSS", title, 1)
    body = {"plan_id": plan["plan_id"], "node": NODE}
    assert client.post("/api/storage/node-reset", json=body).json()["code"] == "CONFIRM_REQUIRED"
    assert client.post("/api/storage/node-reset", json={**body, "confirm": NODE}).json()["code"] == "CONFIRM_REQUIRED"
    done = client.post("/api/storage/node-reset", json={**body, "confirm": f"  {title.upper()} "}).json()
    assert (sorted(done["deleted"]), done["generation"]) == ([NODE, f"{NODE}-a2"], 1)
    assert node_generation(tmp_path, NODE) == 1
    assert client.post("/api/storage/node-reset", json={**body, "confirm": title}).json()["code"] == "PLAN_EXPIRED"


def test_busy_step_has_no_plan_id_and_a_reason(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _attempt(tmp_path, NODE, "running")
    plan = client.get("/api/storage/node-reset", params={"node": NODE}).json()
    assert (plan["plan_id"], plan["blocked_by"]) == (None, "RUN_NOT_FINISHED")
    assert "node" in plan["blocked_params"]


def test_reset_counts_against_the_shared_host_threshold(tmp_path: Path) -> None:
    client = _client(tmp_path, StorageSettings(approval_policy="threshold", approval_threshold_gb=1e-9))
    run = _attempt(tmp_path, NODE)
    plan = client.get("/api/storage/node-reset", params={"node": NODE}).json()
    body = {"plan_id": plan["plan_id"], "node": NODE, "confirm": plan["confirm_text"]}
    resp = client.post("/api/storage/node-reset", json=body)
    assert resp.status_code == 403
    command = resp.json()["error"]["params"]["admin_command"]
    assert command.startswith(f"uv run python scripts/cleanup/node_reset.py --node {NODE} --confirm {NODE}")
    assert run.exists()


@pytest.mark.parametrize("node", ["../etc", "n1", "n1-0123456789ABCDEF", "n1-0123456789abcdef-a2"])
def test_only_a_node_base_name_is_accepted(tmp_path: Path, node: str) -> None:
    assert _client(tmp_path).get("/api/storage/node-reset", params={"node": node}).status_code == 422


@pytest.mark.parametrize(
    ("typed", "ok"),
    [("day model", True), ("DẠY  MODEL", True), ("Dạy model", True), ("dạy mode", False), ("", False), (None, False)],
)
def test_typed_confirmation_ignores_case_accents_and_spaces(typed: str | None, ok: bool) -> None:
    from vlm_pipeline.app.storage_reset import same_text

    assert same_text(typed, "Dạy model") is ok
    assert same_text("đường", "Duong") is True


def test_plan_items_name_their_node_for_the_reset_button(tmp_path: Path) -> None:
    """G-4: the storage panel groups rows of one node by ``node_base``; non-node runs carry null."""
    from vlm_pipeline.artifacts.index import OutputRef

    client = _client(tmp_path)
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    for name in (NODE, f"{NODE}-g1-a2"):
        run = _attempt(tmp_path, name, "completed")
        out = run / "out"
        out.mkdir()
        (out / "w").write_bytes(b"w" * 4096)
        idx.record(f"k-{name}", run_id=name, node_type="vlm/eval.benchmark", run_dir=run,
                   outputs={"o": OutputRef("model", str(out), "h")})  # fmt: skip
    cli_run = tmp_path / "outputs" / "runs" / "20261013-0900-smoke-ab12"
    (cli_run / "checkpoints" / "epoch_0_step_1").mkdir(parents=True)
    (cli_run / "checkpoints" / "epoch_0_step_1" / "w").write_bytes(b"w" * 4096)
    (cli_run / "KEEP").touch()
    (cli_run / "run.json").write_text(json.dumps({"run_id": "x", "status": "completed"}))
    items = client.get("/api/storage/plan").json()["items"]
    by_run = {i["run_id"]: i["node_base"] for i in items}
    assert by_run[NODE] == NODE
    assert by_run[f"{NODE}-g1-a2"] == NODE
    assert by_run["20261013-0900-smoke-ab12"] is None  # a CLI run is not a step of a workflow


@pytest.mark.parametrize(
    ("name", "base"),
    [(NODE, NODE), (f"{NODE}-a3", NODE), (f"{NODE}-g2-a10", NODE), (f"{NODE}-model", NODE),
     ("20261013-0900-smoke-ab12", None), ("../n1-0123456789abcdef", None)],
)  # fmt: skip
def test_node_base_of(name: str, base: str | None) -> None:
    from vlm_pipeline.artifacts.node_reset import node_base_of

    assert node_base_of(name) == base
