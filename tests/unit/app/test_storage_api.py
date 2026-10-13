"""Storage panel API (app/storage.py) and the scheduled cleanup pass (app/storage_scheduler.py, P013 G-2)."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlm_pipeline.app.storage import StorageApi, StorageSettings, _Plans, register_storage_routes
from vlm_pipeline.app.storage_scheduler import LAST_PASS_REL, RetentionScheduler, mode_from_env
from vlm_pipeline.artifacts.index import ArtifactIndex, OutputRef
from vlm_pipeline.artifacts.retention import CLEANUP_LOCK_REL
from vlm_pipeline.artifacts.testing import FakeRunSpec, make_fake_run
from vlm_pipeline.common.app_config import POLICY_REL, policy_path
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.i18n import load as i18n_load

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
    absent = tmp_path / "no-policy.toml"
    assert StorageSettings.from_env(tmp_path, absent).approval_policy == "threshold"  # no profile: fails closed (N3)
    monkeypatch.setenv("VLM_PROFILE", "personal")
    assert StorageSettings.from_env(tmp_path, absent).approval_policy == "none"  # personal install
    monkeypatch.setenv("VLM_PROFILE", "shared_dev")
    shared = StorageSettings.from_env(tmp_path, absent)
    assert (shared.approval_policy, shared.approval_threshold_gb, shared.node_cache_max_gb) == ("threshold", 20.0, 50.0)
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_POLICY", "none")  # cannot relax a shared host
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_THRESHOLD_GB", "500")  # cannot raise the threshold
    assert StorageSettings.from_env(tmp_path, absent) == shared
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_THRESHOLD_GB", "5")
    assert StorageSettings.from_env(tmp_path, absent).approval_threshold_gb == 5.0
    monkeypatch.setenv("VLM_STORAGE_APPROVAL_POLICY", "yolo")
    with pytest.raises(ValueError, match=r"none\|threshold"):
        StorageSettings.from_env(tmp_path, absent)


def test_admin_profile_cannot_be_loosened_by_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VLM_STORAGE_APPROVAL_POLICY", "VLM_STORAGE_APPROVAL_THRESHOLD_GB", "VLM_RETENTION_SCHEDULE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VLM_NODE_CACHE_MAX_GB", "50")
    monkeypatch.setenv("VLM_PROFILE", "personal")
    shared = _policy(tmp_path, '[deployment]\nprofile = "shared_dev"\n')
    assert StorageSettings.from_env(tmp_path, shared).approval_policy == "threshold"
    assert mode_from_env(shared) == "dry_run"
    odd = _policy(tmp_path, '[deployment]\nprofile = "lab"\n')  # unknown admin value -> shared_dev
    assert StorageSettings.from_env(tmp_path, odd).approval_policy == "threshold"
    assert mode_from_env(odd) == "dry_run"
    personal = _policy(tmp_path, '[deployment]\nprofile = "personal"\n')
    assert (StorageSettings.from_env(tmp_path, personal).approval_policy, mode_from_env(personal)) == ("none", "apply")
    monkeypatch.setenv("VLM_PROFILE", "shared_dev")  # env may tighten an admin "personal"
    assert (StorageSettings.from_env(tmp_path, personal).approval_policy, mode_from_env(personal)) == (
        "threshold",
        "dry_run",
    )


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


DAY = 86_400.0


def _aged(path: Path, days: float) -> None:
    ts = time.time() - days * DAY
    for p in [path, *path.rglob("*")]:
        os.utime(p, (ts, ts), follow_symlinks=False)


def _ckpt(run: Path, name: str, days: float, nbytes: int = 4 * KB) -> Path:
    ckpt = run / "checkpoints" / name
    (ckpt / "model").mkdir(parents=True)
    (ckpt / "optim").mkdir()
    (ckpt / "model" / "w.safetensors").write_bytes(b"w" * nbytes)
    (ckpt / "optim" / "o.distcp").write_bytes(b"o" * nbytes)
    _aged(ckpt, days)
    return ckpt


def _retention_runs(root: Path) -> None:
    """r-done: completed, old non-LATEST checkpoint (auto). r-fail: failed 10 days ago, LATEST (manual, permanent).
    r-keep: KEEP (kept). node n1: set-aside after a failed retry (manual, TRAINED_MODEL)."""
    runs = root / "outputs" / "runs"
    done = runs / "r-done"
    _ckpt(done, "epoch_0_step_10", 10)
    _ckpt(done, "epoch_0_step_20", 1)
    (done / "checkpoints" / "LATEST").symlink_to("epoch_0_step_20")
    (done / "run.json").write_text(json.dumps({"run_id": "r-done", "status": "completed"}))
    fail = runs / "r-fail"
    _ckpt(fail, "epoch_0_step_10", 10)
    (fail / "checkpoints" / "LATEST").symlink_to("epoch_0_step_10")
    _aged(fail / "checkpoints", 10)  # the LATEST link counts as activity too
    (fail / "run.json").write_text(
        json.dumps({"run_id": "r-fail", "status": "failed", "ended_at": time.time() - 10 * DAY})
    )
    _aged(fail / "run.json", 10)
    keep = runs / "r-keep"
    _ckpt(keep, "epoch_0_step_10", 10)
    (keep / "run.json").write_text(json.dumps({"run_id": "r-keep", "status": "completed"}))
    (keep / "KEEP").touch()
    node = root / "outputs" / "graph" / "runs" / "n1"
    node.mkdir(parents=True)
    (node / "run.json").write_text(json.dumps({"run_id": "n1", "status": "failed", "stage": "train"}))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() - 5 * DAY))
    aside = node / f"checkpoints.not-resumable-{stamp}"
    (aside / "epoch_0_step_9").mkdir(parents=True)
    (aside / "epoch_0_step_9" / "w").write_bytes(b"w" * 4 * KB)
    _aged(aside, 5)


def _plan_items(client: TestClient) -> tuple[str, dict[str, dict[str, object]]]:
    body = client.get("/api/storage/plan").json()
    assert body["schema_version"] == 3
    return body["plan_id"], {i["item_id"]: i for i in body["items"]}


def test_plan_merges_checkpoint_and_set_aside_items_by_section(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, _, root, _ = env
    _retention_runs(root)
    _, items = _plan_items(client)
    aside_id = next(k for k in items if k.startswith("graph:n1/"))
    got = {k: (i["source"], i["section"], i["consequence_code"], i["confirm_text"]) for k, i in items.items()}
    assert got == {
        "r-done/checkpoints/epoch_0_step_10": ("checkpoint", "auto", None, None),
        "r-fail/checkpoints/epoch_0_step_10": ("checkpoint", "manual", "PERMANENT_LOSS", "r-fail"),
        "r-keep/checkpoints/epoch_0_step_10": ("checkpoint", "kept", None, None),
        aside_id: ("set_aside", "manual", "PERMANENT_LOSS", "n1"),
    }
    assert items[aside_id]["protected_reason"] == "TRAINED_MODEL"
    assert items["r-keep/checkpoints/epoch_0_step_10"]["lock_reason"] == "PINNED_OR_KEEP"
    assert items["r-keep/checkpoints/epoch_0_step_10"]["size_gb"] == round(8 * KB / GIB, 3)  # kept items show size


def test_apply_retention_items_needs_typed_confirmation_for_permanent_loss(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, _, root, _ = env
    _retention_runs(root)
    plan_id, items = _plan_items(client)
    aside_id = next(k for k in items if k.startswith("graph:n1/"))
    ids = ["r-done/checkpoints/epoch_0_step_10", "r-fail/checkpoints/epoch_0_step_10", aside_id]
    body = client.post(
        "/api/storage/apply", json={"plan_id": plan_id, "item_ids": [*ids, "r-keep/checkpoints/epoch_0_step_10"]}
    ).json()
    assert [i["item_id"] for i in body["pruned"]] == ["r-done/checkpoints/epoch_0_step_10"]
    assert {s["item_id"]: s["reason"] for s in body["skipped"]} == {
        "r-fail/checkpoints/epoch_0_step_10": "CONFIRM_REQUIRED",
        aside_id: "CONFIRM_REQUIRED",
        "r-keep/checkpoints/epoch_0_step_10": "NOT_IN_PLAN",  # kept items are never part of a plan
    }
    plan_id, _ = _plan_items(client)
    confirm = {ids[1]: "r-fail", aside_id: "n1"}
    body = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ids[1:], "confirm": confirm}).json()
    assert sorted(i["item_id"] for i in body["pruned"]) == sorted(ids[1:])
    assert not (root / "outputs" / "runs" / "r-fail" / "checkpoints" / "epoch_0_step_10").exists()
    assert (root / "outputs" / "runs" / "r-keep" / "checkpoints" / "epoch_0_step_10").exists()
    events = [json.loads(line) for line in (root / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()]
    assert {(e["by"], e["manual"]) for e in events} == {("app-user", True)}


def test_threshold_counts_checkpoints_and_cache_together(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    settings = StorageSettings(approval_policy="threshold", approval_threshold_gb=4 * KB / GIB, node_cache_max_gb=0.0)
    app = FastAPI()
    register_storage_routes(app, StorageApi(tmp_path, idx, settings))
    client = TestClient(app)
    _retention_runs(tmp_path)
    plan_id, _ = _plan_items(client)
    ids = ["r-done/checkpoints/epoch_0_step_10"]  # 8 KB > 4 KB threshold
    resp = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": ids})
    assert resp.status_code == 403
    command = resp.json()["error"]["params"]["admin_command"]
    assert "scripts/cleanup/retention.py --items r-done/checkpoints/epoch_0_step_10" in command
    assert "--approved-by" in command
    assert (tmp_path / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()


def _interrupted(root: Path, run_id: str, days_ago: float, name: str = "") -> None:
    run = root / "outputs" / "runs" / run_id
    _ckpt(run, "epoch_0_step_10", days_ago)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_10")
    (run / "run.json").write_text(
        json.dumps({"run_id": run_id, "name": name, "status": "interrupted", "ended_at": time.time() - days_ago * DAY})
    )
    _aged(run, days_ago)


def test_interrupted_run_is_kept_until_a_date_and_warned_a_day_before(
    env: tuple[TestClient, ArtifactIndex, Path, Clock],
) -> None:
    client, _, root, _ = env
    _interrupted(root, "r-soon", 6.5, name="train.lora_vlm Qwen3.5 2B")  # ~12 h left; app runs: "<node type> <entry>"
    _interrupted(root, "r-script", 6.6, name="smoke p3b")  # not a node: its name is shown
    _interrupted(root, "r-fresh", 1)  # 6 days left
    body = client.get("/api/storage/plan", params={"locale": "vi"}).json()
    items = {i["item_id"]: i for i in body["items"]}
    soon = items["r-soon/checkpoints/epoch_0_step_10"]
    assert (soon["section"], soon["lock_reason"]) == ("kept", "RUN_INTERRUPTED")
    assert set(soon["lock_params"]) == {"run", "until"}
    assert items["r-fresh/checkpoints/epoch_0_step_10"]["section"] == "kept"
    script, soon = body["attention"]
    title = i18n_load("vi")["node.train.lora_vlm.title"]
    assert (soon["code"], soon["run_id"], soon["node"]) == ("RUN_INTERRUPTED_EXPIRING", "r-soon", title)
    assert f'"{title}"' in soon["message"]
    assert '"r-soon"' in soon["message"]
    assert soon["resume_until"][11:16] in soon["message"]  # same HH:MM as the date the panel shows
    assert script["node"] == "smoke p3b"


def test_run_name_in_a_notice_is_a_clean_label(env: tuple[TestClient, ArtifactIndex, Path, Clock]) -> None:
    """A job can write its own run.json: its name reaches the panel without bidi/control characters, shortened."""
    client, _, root, _ = env
    _interrupted(root, "r-x", 6.5, name="\u202eevil\x07" + "a" * 5000)
    (notice,) = client.get("/api/storage/plan", params={"locale": "en"}).json()["attention"]
    assert notice["node"].startswith("evila")
    assert len(notice["node"]) <= 120
    assert "\u202e" not in notice["message"]


def test_expired_interrupted_run_latest_needs_a_person(env: tuple[TestClient, ArtifactIndex, Path, Clock]) -> None:
    client, _, root, _ = env
    _interrupted(root, "r-old", 8)
    body = client.get("/api/storage/plan", params={"locale": "en"}).json()
    (item,) = [i for i in body["items"] if i["run_id"] == "r-old"]
    assert (item["section"], item["consequence_code"], item["confirm_text"]) == ("manual", "PERMANENT_LOSS", "r-old")
    assert body["attention"] == []


@pytest.mark.parametrize("locale", ["vi", "en"])
def test_every_retention_lock_reason_and_item_has_copy(locale: str) -> None:
    from vlm_pipeline.i18n import load

    texts = load(locale)
    for reason in ("PINNED_OR_KEEP", "GPU_LOCK_HOLDER", "UNKNOWN_STATUS", "RUN_NOT_FINISHED", "RUN_INTERRUPTED"):
        assert f"storage.lock_reason.{reason}" in texts
    assert "storage.attention.RUN_INTERRUPTED_EXPIRING" in texts
    for key in ("checkpoint_of_run", "set_aside_of_run"):
        assert f"storage.item.{key}" in texts


# --- scheduled pass (G-2) ---
AUTO = "r-done/checkpoints/epoch_0_step_10"
PERMANENT = "r-fail/checkpoints/epoch_0_step_10"


def _scheduler(root: Path, settings: StorageSettings | None = None) -> tuple[RetentionScheduler, ArtifactIndex]:
    (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(root / "outputs" / "graph" / "app.db")
    settings = settings or StorageSettings(node_cache_max_gb=0.0)
    return RetentionScheduler(root, idx, settings, lock_path=root / ".team" / "GPU_LOCK"), idx


def test_every_pass_reconciles_stale_runs_in_any_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from functools import partial

    from vlm_pipeline.artifacts import stale_runs
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest, write_manifest

    monkeypatch.setattr(stale_runs, "reconcile", partial(stale_runs.reconcile, containers=set))  # no docker CLI call
    s, _ = _scheduler(tmp_path)
    s.mode = "off"
    stuck = tmp_path / "outputs" / "runs" / "probe"
    write_manifest(stuck, RunManifest(run_id="probe", name="probe", stage="eval", task="x", status=RunStatus.RUNNING))
    old = time.time() - 3 * stale_runs.QUIET_S
    for path in (stuck, *stuck.iterdir()):
        os.utime(path, (old, old))
    assert s.run_once() == {"status": "off", "reconciled": ["probe"]}
    assert read_manifest(stuck).status == RunStatus.INTERRUPTED
    s.mode = "dry_run"
    assert s.run_once()["reconciled"] == []  # once
    s.lock_path.parent.mkdir(parents=True, exist_ok=True)
    s.lock_path.write_text("not json")  # a malformed lock: no run is judged, the pass still runs
    assert s.run_once()["reconciled"] == []


def test_apply_pass_does_only_safe_items(tmp_path: Path) -> None:
    s, idx = _scheduler(tmp_path)
    s.mode = "apply"
    run = _cached(tmp_path, idx, "eval-1")
    _cached(tmp_path, idx, "train-1", train=True)
    _retention_runs(tmp_path)
    summary = s.run_once()
    assert summary["status"] == "applied"
    assert summary["cache_evict"] == ["eval-1"]  # the trained model is protected
    assert summary["retention"] == [AUTO]
    assert PERMANENT in summary["left_to_a_person"]
    assert any(i.startswith("graph:n1/") for i in summary["left_to_a_person"])
    assert not (run / "checkpoints").exists()
    assert not (tmp_path / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()
    assert (tmp_path / "outputs" / "runs" / "r-fail" / "checkpoints" / "epoch_0_step_10").exists()
    assert (tmp_path / "outputs" / "runs" / "train-1" / "checkpoints").exists()
    events = [json.loads(x) for x in (tmp_path / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()]
    assert {(e["by"], e["manual"]) for e in events} == {("app-scheduler", False)}
    assert json.loads((tmp_path / LAST_PASS_REL).read_text())["status"] == "applied"


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    s, idx = _scheduler(tmp_path)
    run = _cached(tmp_path, idx, "eval-1")
    _retention_runs(tmp_path)
    summary = s.run_once()
    assert summary["status"] == "dry_run"
    assert summary["retention"] == [AUTO]
    assert (run / "checkpoints").exists()
    assert (tmp_path / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()
    assert not (tmp_path / "outputs" / "system" / "cleanup.jsonl").exists()


@pytest.mark.parametrize("lock_text", ['{"who": "x"}', "garbage"])
def test_any_gpu_lock_file_skips_the_pass(tmp_path: Path, lock_text: str) -> None:
    s, _ = _scheduler(tmp_path)
    s.mode = "apply"
    _retention_runs(tmp_path)
    (tmp_path / ".team").mkdir()
    (tmp_path / ".team" / "GPU_LOCK").write_text(lock_text)
    assert s.run_once()["reason"] == "GPU_LOCK_HELD"
    assert (tmp_path / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()


def test_threshold_quota_defers_what_does_not_fit(tmp_path: Path) -> None:
    settings = StorageSettings(approval_policy="threshold", approval_threshold_gb=4 * KB / GIB, node_cache_max_gb=0.0)
    s, _ = _scheduler(tmp_path, settings)
    s.mode = "apply"
    _retention_runs(tmp_path)  # AUTO is 8 KB > 4 KB
    summary = s.run_once()
    assert summary["retention"] == []
    assert summary["retention_deferred_by_quota"] == [AUTO]
    assert (tmp_path / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()


def test_concurrent_pass_is_skipped(tmp_path: Path) -> None:
    s, _ = _scheduler(tmp_path)
    lock = tmp_path / CLEANUP_LOCK_REL
    lock.parent.mkdir(parents=True)
    with lock.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert s.run_once()["reason"] == "PASS_IN_PROGRESS"


def test_mode_defaults_follow_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    absent = tmp_path / "no-policy.toml"
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE", raising=False)
    monkeypatch.delenv("VLM_PROFILE", raising=False)
    assert mode_from_env(absent) == "dry_run"  # unset profile fails closed
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "apply")
    assert mode_from_env(absent) == "dry_run"
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE")
    monkeypatch.setenv("VLM_PROFILE", "shared_dev")
    assert mode_from_env(absent) == "dry_run"
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "apply")
    assert mode_from_env(absent) == "dry_run"  # env may only tighten on a shared host (D17)
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "off")
    assert mode_from_env(absent) == "off"
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE")
    monkeypatch.setenv("VLM_PROFILE", "personal")
    assert mode_from_env(absent) == "apply"
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "off")
    assert mode_from_env(absent) == "off"
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "yes")
    with pytest.raises(ValueError, match=r"off\|dry_run\|apply"):
        mode_from_env(absent)


def _policy(tmp_path: Path, text: str, mode: int = 0o644) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_admin_policy_turns_apply_on_for_a_shared_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE", raising=False)
    monkeypatch.setenv("VLM_PROFILE", "shared_dev")
    policy = _policy(tmp_path, '[deployment]\nprofile = "shared_dev"\n[storage]\nschedule = "apply"\n')
    assert mode_from_env(policy) == "apply"
    monkeypatch.setenv("VLM_RETENTION_SCHEDULE", "dry_run")
    assert mode_from_env(policy) == "dry_run"  # env may still lower it
    monkeypatch.setenv("VLM_PROFILE", "personal")
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE")
    assert mode_from_env(_policy(tmp_path, '[storage]\nschedule = "off"\n')) == "off"  # admin also lowers


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ('[storage]\nschedule = "apply"\n', 0o666),  # writable by others
        ('[storage]\nschedule = "yes"\n', 0o644),  # unknown value
        ("[storage\nschedule = apply", 0o644),  # not TOML
        ("[storage]\n", 0o644),  # no key
    ],
)
def test_unusable_admin_policy_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str, mode: int
) -> None:
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE", raising=False)
    monkeypatch.delenv("VLM_PROFILE", raising=False)
    assert mode_from_env(_policy(tmp_path, text, mode)) == "dry_run"


def test_symlinked_admin_policy_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE", raising=False)
    monkeypatch.delenv("VLM_PROFILE", raising=False)
    real = _policy(tmp_path, '[storage]\nschedule = "apply"\n')
    link = tmp_path / "link.toml"
    link.symlink_to(real)
    assert mode_from_env(link) == "dry_run"


def test_admin_command_quotes_item_ids_from_directory_names(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    idx = ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db")
    settings = StorageSettings(approval_policy="threshold", approval_threshold_gb=1 / GIB, node_cache_max_gb=0.0)
    app = FastAPI()
    register_storage_routes(app, StorageApi(tmp_path, idx, settings))
    client = TestClient(app)
    run = tmp_path / "outputs" / "runs" / "x;touch pwned"
    _ckpt(run, "epoch_0_step_10", 10)
    _ckpt(run, "epoch_0_step_20", 1)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_20")
    (run / "run.json").write_text(json.dumps({"run_id": "x", "status": "completed"}))
    plan_id, _ = _plan_items(client)
    item = "x;touch pwned/checkpoints/epoch_0_step_10"
    resp = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": [item]})
    command = resp.json()["error"]["params"]["admin_command"]
    assert f"--items '{item}' --apply" in command


def test_panel_apply_waits_for_a_running_cleanup_pass(env: tuple[TestClient, ArtifactIndex, Path, Clock]) -> None:
    client, _, root, _ = env
    _retention_runs(root)
    plan_id, _ = _plan_items(client)
    lock = root / CLEANUP_LOCK_REL
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        resp = client.post("/api/storage/apply", json={"plan_id": plan_id, "item_ids": [AUTO]})
    assert resp.status_code == 409
    assert resp.json()["code"] == "CLEANUP_IN_PROGRESS"
    assert (root / "outputs" / "runs" / "r-done" / "checkpoints" / "epoch_0_step_10").exists()


def test_last_pass_file_is_not_written_through_a_planted_symlink(tmp_path: Path) -> None:
    s, _ = _scheduler(tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    tmp = (tmp_path / LAST_PASS_REL).with_suffix(".tmp")
    tmp.parent.mkdir(parents=True)
    tmp.symlink_to(victim)
    s.run_once()
    assert victim.read_text() == "keep me"
    assert json.loads((tmp_path / LAST_PASS_REL).read_text())["status"] == "dry_run"


def test_cleanup_lock_refuses_a_symlink(tmp_path: Path) -> None:
    from vlm_pipeline.artifacts.retention import cleanup_lock

    lock = tmp_path / CLEANUP_LOCK_REL
    lock.parent.mkdir(parents=True)
    lock.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError, match="Too many levels of symbolic links"), cleanup_lock(tmp_path):
        pass


def test_policy_path_ignores_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    (fake_home / POLICY_REL).parent.mkdir(parents=True)
    (fake_home / POLICY_REL).write_text('[storage]\nschedule = "apply"\n')
    monkeypatch.setenv("HOME", str(fake_home))
    assert not policy_path().is_relative_to(fake_home)  # from the passwd database, not $HOME


def test_policy_in_a_group_writable_dir_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_RETENTION_SCHEDULE", raising=False)
    monkeypatch.delenv("VLM_PROFILE", raising=False)
    policy = _policy(tmp_path, '[storage]\nschedule = "apply"\n')
    tmp_path.chmod(0o775)
    assert mode_from_env(policy) == "dry_run"
    tmp_path.chmod(0o755)
    assert mode_from_env(policy) == "apply"


def test_kept_size_is_zero_for_a_path_outside_its_runs_root(tmp_path: Path) -> None:
    from vlm_pipeline.artifacts.retention import PlanItem

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    api = StorageApi(tmp_path, ArtifactIndex(tmp_path / "outputs" / "graph" / "app.db"), StorageSettings())
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "big").write_bytes(b"x" * 4 * KB)
    run = tmp_path / "outputs" / "runs" / "r1" / "checkpoints"
    run.mkdir(parents=True)
    (run / "epoch_0_step_1").symlink_to(outside)
    item = PlanItem(
        "r1", "checkpoints/epoch_0_step_1", "checkpoint", (), "keep", 0, "D7.blocked", "x", "PINNED_OR_KEEP"
    )
    assert api._kept_bytes(item) == 0
