"""MODEL-005 quick trial ("Chạy thử nhanh", P021 §4): a run option on POST /api/runs, never a workflow param."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.app.test_api import FakeServices
from tests.unit.app.test_user_assets import _stage

from vlm_pipeline.app import user_assets
from vlm_pipeline.app.main import create_app
from vlm_pipeline.graph.trial import TRIAL_STEPS, TrialNotPossibleError, trial_workflow
from vlm_pipeline.nodes.train import TRIAL_CAPABLE


def _n(i: str, t: str, **p: Any) -> dict[str, Any]:
    return {"id": i, "type": t, "version": "1.0.0", "params": p}


def _e(a: str, out: str, b: str, port: str) -> dict[str, list[str]]:
    return {"from": [a, out], "to": [b, port]}


def _wf(user_model: str = "") -> dict[str, Any]:
    pick = _n("m", "model.hf_pick", **({"user_model": user_model} if user_model else {}))
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t",
            "nodes": [pick, _n("d", "data.hf_load"), _n("t", "train.lora_vlm"), _n("g", "train.merge_lora"),
                      _n("x", "export.hf_dir"), _n("o", "data.hf_load"), _n("r", "train.dpo_vlm")],
            "edges": [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data"),
                      _e("m", "model", "g", "base_model"), _e("t", "adapter", "g", "adapter"),
                      _e("g", "model", "x", "model")]}  # fmt: skip


def test_the_trial_keeps_the_step_and_its_ancestors_only() -> None:
    cut = trial_workflow(_wf(), "t", TRIAL_CAPABLE)
    assert sorted(n["id"] for n in cut["nodes"]) == ["d", "m", "t"]  # no merge/export, no unrelated branch
    assert cut["edges"] == [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data")]
    assert len(_wf()["nodes"]) == 7  # the saved workflow is never changed


@pytest.mark.parametrize("node", ["g", "nope", "r"])
def test_only_a_training_step_has_a_trial(node: str) -> None:
    with pytest.raises(TrialNotPossibleError):
        trial_workflow(_wf(), node, TRIAL_CAPABLE)


def test_post_runs_takes_a_trial_and_marks_its_events(tmp_path: Path) -> None:
    svc = FakeServices(tmp_path)
    svc.gate.set()
    client = TestClient(create_app(svc, secure=False))
    r = client.post("/api/runs", json={"workflow": _wf(), "trial": {"node": "t"}})
    assert r.status_code == 202
    graph_run_id = r.json()["graph_run_id"]
    time.sleep(0.05)
    workflow, trial = svc.started
    assert (sorted(n["id"] for n in workflow["nodes"]), trial) == (["d", "m", "t"], {"t": TRIAL_STEPS})
    with client.websocket_connect(f"/api/runs/{graph_run_id}/events") as ws:
        events = [ws.receive_json() for _ in range(5)]
    assert events[-1]["type"] == "run_status"
    assert events[-1]["trial"] is True
    assert all("trial" not in e for e in events if e["type"] == "node_status")


def test_a_bare_workflow_is_a_normal_run(tmp_path: Path) -> None:
    svc = FakeServices(tmp_path)
    svc.gate.set()
    client = TestClient(create_app(svc, secure=False))
    assert client.post("/api/runs", json=_wf()).status_code == 202
    time.sleep(0.05)
    assert svc.started[1] is None
    refused = client.post("/api/runs", json={"workflow": _wf(), "trial": {"node": "g"}})
    assert (refused.status_code, refused.json()["code"]) == (400, "CFG-005")


def test_a_finished_trial_is_kept_on_the_user_models_record(tmp_path: Path) -> None:
    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    user_assets.save(tmp_path, record)
    model_id = record.entry.id
    assert user_assets.picked_user_models(_wf(model_id), "t") == [model_id]
    assert user_assets.picked_user_models(_wf(model_id), "x") == []  # export reads the merge, not the pick

    class Done(FakeServices):
        def run_graph(self, workflow: Any, graph_run_id: str, on_event: Any, trial_max_steps: Any = None) -> None:
            on_event({"type": "node_status", "node_id": "t", "status": "completed"})
            on_event({"type": "run_status", "status": "completed"})

    client = TestClient(create_app(Done(tmp_path), secure=False))
    graph_run_id = client.post("/api/runs", json={"workflow": _wf(model_id), "trial": {"node": "t"}}).json()[
        "graph_run_id"]  # fmt: skip
    deadline = time.time() + 5
    while (saved := user_assets.load(tmp_path, model_id)) is not None and saved.trial_run_id is None:
        assert time.time() < deadline
        time.sleep(0.02)
    assert saved is not None
    assert saved.trial_run_id == graph_run_id


def test_validate_hints_a_trial_until_one_finished(tmp_path: Path) -> None:
    from vlm_pipeline.app.services import DefaultServices

    record = user_assets.finish_model(tmp_path, _stage(tmp_path / "run"), "o/x")
    user_assets.save(tmp_path, record)
    from vlm_pipeline.nodes import REGISTRY

    services = DefaultServices(registry=REGISTRY, root=tmp_path)
    hints = services._untried_user_models(_wf(record.entry.id))
    assert [(h.code, h.severity, h.node_id, dict(h.params)) for h in hints] == [  # type: ignore[attr-defined]
        ("MODEL-005", "info", "t", {"node": "t", "model": "o/x"})
    ]
    user_assets.save(tmp_path, record.model_copy(update={"trial_run_id": "g-1"}))
    assert services._untried_user_models(_wf(record.entry.id)) == []


def test_a_worker_never_runs_a_full_job_in_a_trials_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """bao-security: a step whose engine has no trial cap refuses the trial instead of training in full."""
    from vlm_pipeline.app.workers import NodeNotRunnableError, RealWorker
    from vlm_pipeline.graph.worker import NodeJob
    from vlm_pipeline.nodes import REGISTRY

    worker = RealWorker(tmp_path)
    ran: list[str] = []
    monkeypatch.setattr(worker, "check_support", lambda query: None)
    monkeypatch.setattr(worker, "support_query", lambda job: None)
    monkeypatch.setitem(worker._handlers, "train.dpo_vlm", lambda job: ran.append(job.node_id))  # type: ignore[arg-type,return-value]
    spec = REGISTRY.get("train.dpo_vlm")
    assert spec is not None
    job = NodeJob("r", spec, spec.validate_params({}), {}, tmp_path, trial_max_steps=TRIAL_STEPS)
    with pytest.raises(NodeNotRunnableError):
        worker.run(job)
    assert ran == []


def test_user_assets_carry_licence_meta_like_the_catalog(tmp_path: Path) -> None:
    """bao-security: license_class + obligations from the obligations table; an id it lacks is unknown (fail-closed)."""
    from vlm_pipeline.common.obligations import licence_meta

    known = user_assets.finish_model(tmp_path, _stage(tmp_path / "a"), "o/x").to_api()
    assert known["license_class"] == "commercial_ok"
    assert known["obligations"]
    unknown = user_assets.finish_model(tmp_path, _stage(tmp_path / "b", license_id="other"), "o/x").to_api()
    assert unknown["license_class"] == "unknown"
    assert licence_meta("my-own-terms", "dataset")["license_class"] == "unknown"
    assert licence_meta("mit", "dataset")["license_class"] != "unknown"
