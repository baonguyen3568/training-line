"""Storage panel API: plan (dry-run) -> apply, protections, approval policy (app/storage.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlm_pipeline.app.storage import StorageApi, StorageSettings, _Plans, register_storage_routes
from vlm_pipeline.artifacts.index import ArtifactIndex, OutputRef
from vlm_pipeline.artifacts.testing import FakeRunSpec, make_fake_run

KB = 1024


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _cached(root: Path, idx: ArtifactIndex, run_id: str, *, train: bool = False) -> Path:
    stage, task = ("train", "lora") if train else ("eval", "eval")
    run = make_fake_run(root / "outputs" / "runs", FakeRunSpec(run_id, stage=stage, task=task, checkpoint_every=10))  # type: ignore[arg-type]
    idx.record(
        run_id,
        run_id=run_id,
        node_type="vlm/train.lora" if train else "vlm/eval.benchmark",
        outputs={"out": OutputRef("model", str(run / "checkpoints"), "h")},
        trained_model=train,
        run_dir=run,
    )
    return run


@pytest.fixture
def env(tmp_path: Path) -> tuple[TestClient, ArtifactIndex, Path, Clock]:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    clock = Clock()
    api = StorageApi(tmp_path, idx, StorageSettings(node_cache_max_gb=0.0), plans=_Plans(clock=clock))
    app = FastAPI()
    register_storage_routes(app, api)
    return TestClient(app), idx, tmp_path, clock


def test_plan_lists_items_and_reports_disk003(env: tuple[TestClient, ArtifactIndex, Path, Clock]) -> None:
    client, idx, root, _ = env
    _cached(root, idx, "eval-1")
    _cached(root, idx, "train-1", train=True)
    body = client.get("/api/storage/plan", params={"locale": "en"}).json()
    items = {i["item_id"]: i for i in body["items"]}
    assert items["eval-1"]["auto"] is True
    assert items["eval-1"]["confirm_required"] is False
    assert items["train-1"]["auto"] is False
    assert items["train-1"]["protected_reason"] == "TRAINED_MODEL"
    assert items["train-1"]["confirm_text"] == "train-1"
    assert items["train-1"]["consequence_code"] == "PERMANENT_LOSS"
    assert body["error"]["code"] == "DISK-003"  # cap 0 cannot be met without the trained model
    assert body["approval_policy"] == "none"
    assert body["threshold_gb"] is None


def test_apply_evicts_regenerable_and_needs_confirm_for_protected(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, idx, root, _ = env
    ev = _cached(root, idx, "eval-1")
    tr = _cached(root, idx, "train-1", train=True)
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    res = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-1", "train-1"]}).json()
    assert [e["cache_key"] for e in res["evicted"]] == ["eval-1"]
    assert {"item_id": "train-1", "reason": "CONFIRM_REQUIRED"} in res["skipped"]
    assert not (ev / "checkpoints").exists()
    assert (tr / "checkpoints").exists()
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    res = client.post(
        "/api/storage/apply",
        json={"plan_id": plan_id, "item_ids": ["train-1"], "confirm": {"train-1": "train-1"}},
    ).json()
    assert [e["cache_key"] for e in res["evicted"]] == ["train-1"]
    assert not (tr / "checkpoints").exists()


def test_items_outside_the_plan_and_expired_plans_are_refused(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, idx, root, clock = env
    run = _cached(root, idx, "eval-1")
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    _cached(root, idx, "eval-2")  # appeared after the user looked at the plan
    res = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-2", "../../etc"]}).json()
    assert res["evicted"] == []
    assert {s["reason"] for s in res["skipped"]} == {"NOT_IN_PLAN"}
    clock.t += 601
    expired = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-1"]})
    assert expired.status_code == 409
    assert (run / "checkpoints").exists()


def test_threshold_policy_answers_disk002_and_deletes_nothing(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    run = _cached(tmp_path, idx, "eval-1")
    settings = StorageSettings(approval_policy="threshold", approval_threshold_gb=0.0, node_cache_max_gb=0.0)
    app = FastAPI()
    register_storage_routes(app, StorageApi(tmp_path, idx, settings))
    client = TestClient(app)
    plan = client.get("/api/storage/plan").json()
    assert plan["threshold_gb"] == 0.0
    res = client.post("/api/storage/apply", json={"plan_id": plan["plan_id"], "item_ids": ["eval-1"]})
    assert res.status_code == 403
    body = res.json()
    assert body["deleted"] == []
    assert body["error"]["code"] == "DISK-002"
    assert "--items eval-1" in body["error"]["params"]["admin_command"]
    assert (run / "checkpoints").exists()


def test_settings_follow_profile_and_env_only_tightens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VLM_PROFILE", "VLM_STORAGE_APPROVAL_POLICY", "VLM_STORAGE_APPROVAL_THRESHOLD_GB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VLM_NODE_CACHE_MAX_GB", "50")
    assert StorageSettings.from_env(tmp_path).approval_policy == "none"  # personal install
    monkeypatch.setenv("VLM_PROFILE", "shared_dev")
    shared = StorageSettings.from_env(tmp_path)
    assert (shared.approval_policy, shared.approval_threshold_gb, shared.node_cache_max_gb) == ("threshold", 20.0, 50.0)
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_POLICY", "none")  # cannot relax a shared host
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_THRESHOLD_GB", "500")  # cannot raise the threshold
    assert StorageSettings.from_env(tmp_path) == shared
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_THRESHOLD_GB", "5")
    assert StorageSettings.from_env(tmp_path).approval_threshold_gb == 5.0
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_POLICY", "yolo")
    with pytest.raises(ValueError, match=r"none\|threshold"):
        StorageSettings.from_env(tmp_path)


def test_small_manual_batches_still_need_approval_on_shared_host(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    a = _cached(tmp_path, idx, "eval-a")
    b = _cached(tmp_path, idx, "eval-b")
    api = StorageApi(tmp_path, idx, StorageSettings(node_cache_max_gb=1e9))
    app = FastAPI()
    register_storage_routes(app, api)
    client = TestClient(app)
    from vlm_pipeline.artifacts.cache_retention import list_cache

    biggest = max(item.reclaim_bytes for item, _ in list_cache(tmp_path, idx)) / 1024**3
    # threshold between one item and two items: each request alone is under it
    api.settings = StorageSettings(
        approval_policy="threshold", approval_threshold_gb=1.5 * biggest, node_cache_max_gb=1e9
    )
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    first = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-a"], "confirm": {}})
    assert first.status_code == 200
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    second = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-b"]})
    assert second.status_code == 403
    assert second.json()["error"]["code"] == "DISK-002"
    assert not (a / "checkpoints").exists()
    assert (b / "checkpoints").exists()


def test_item_that_became_protected_after_listing_is_skipped(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, idx, root, _ = env
    run = _cached(root, idx, "eval-1")
    plan_id = client.get("/api/storage/plan").json()["plan_id"]
    (run / "run.json").write_text("{broken")  # now MANIFEST_UNREADABLE, the user never confirmed it
    res = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ["eval-1"]}).json()
    assert res["evicted"] == []
    assert {"item_id": "eval-1", "reason": "CONFIRM_REQUIRED"} in res["skipped"]
    assert (run / "checkpoints").exists()


@pytest.mark.parametrize("locale", ["vi", "en"])
def test_every_protected_reason_and_storage_error_has_copy(locale: str) -> None:
    from vlm_pipeline.artifacts.cache_retention import MANIFEST_UNREADABLE, TRAINED_MODEL
    from vlm_pipeline.i18n import load

    texts = load(locale)
    for code in (TRAINED_MODEL, MANIFEST_UNREADABLE):
        assert f"storage.protected_reason.{code}" in texts
    for code in ("DISK-002", "DISK-003"):
        assert f"error.{code}.title" in texts
        assert f"error.{code}.what" in texts
