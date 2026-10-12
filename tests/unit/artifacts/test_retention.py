"""Unit tests for ``vlm_pipeline.artifacts.retention`` on fake run directories (tmp_path only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.retention import RetentionPolicy, apply, plan
from vlm_pipeline.common.diskguard import LockStatus
from vlm_pipeline.common.paths import GIB

NOW = 1_800_000_000.0
DAY = 86_400.0


def no_lock(_path: Path) -> LockStatus | None:
    return None


def _ckpt(run: Path, name: str, *, age_days: float, optim_bytes: int = 1000, incomplete: bool = False) -> Path:
    ckpt = run / "checkpoints" / name
    for sub in ("model", "optim", "dataloader", "rng"):
        (ckpt / sub).mkdir(parents=True)
    (ckpt / "model" / "adapter_model.safetensors").write_bytes(b"w" * 5000)
    (ckpt / "optim" / "__0_0.distcp").write_bytes(b"o" * optim_bytes)
    (ckpt / "optim" / ".metadata").write_bytes(b"m")
    if incomplete:
        (ckpt / ".incomplete").touch()
    ts = NOW - age_days * DAY
    os.utime(ckpt, (ts, ts))
    return ckpt


def _run(root: Path, run_id: str, *, status: str | None = "completed") -> Path:
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    run = root / "outputs" / "runs" / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "checkpoints" / "training.jsonl").write_text("{}\n")
    if status is not None:
        (run / "run.json").write_text(json.dumps({"run_id": run_id, "status": status}))
    return run


def _actions(root: Path, **kw: object) -> dict[str, tuple[str, str | None]]:
    p = plan(root, now=NOW, lock_reader=no_lock, **kw)  # type: ignore[arg-type]
    return {i.path.split("/")[-1]: (i.action, i.blocked_by) for r in p.runs for i in r.items}


def _standard_run(root: Path, run_id: str = "r1", **kw: object) -> Path:
    run = _run(root, run_id, **kw)  # type: ignore[arg-type]
    _ckpt(run, "epoch_0_step_10", age_days=10)
    _ckpt(run, "epoch_0_step_20", age_days=10)
    _ckpt(run, "epoch_0_step_30", age_days=10)
    _ckpt(run, "epoch_0_step_40", age_days=1)
    _ckpt(run, "epoch_0_step_50", age_days=10, incomplete=True)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_30")
    (run / "checkpoints" / "LOWEST_VAL").symlink_to("epoch_0_step_20")
    return run


def test_rules_for_completed_run(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    assert _actions(tmp_path) == {
        "epoch_0_step_10": ("delete", None),  # other, older than 7 days
        "epoch_0_step_20": ("strip_optim", None),  # LOWEST_VAL
        "epoch_0_step_30": ("strip_optim", None),  # LATEST older than 7 days
        "epoch_0_step_40": ("keep", None),  # other, too young
        "epoch_0_step_50": ("delete", None),  # .incomplete
    }


def test_latest_and_best_same_checkpoint_latest_wins(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1")
    _ckpt(run, "epoch_0_step_10", age_days=1)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_10")
    (run / "checkpoints" / "LOWEST_VAL").symlink_to("epoch_0_step_10")
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    (item,) = p.runs[0].items
    assert item.roles == ("LATEST", "LOWEST_VAL")
    assert item.action == "keep"  # LATEST rule (kept whole for 7 days) beats best-strip


@pytest.mark.parametrize("status", ["running", "failed", "interrupted", "killed"])
def test_unfinished_runs_are_never_touched(tmp_path: Path, status: str) -> None:
    _standard_run(tmp_path, status=status)
    assert set(_actions(tmp_path).values()) == {("keep", "RUN_NOT_FINISHED")}


@pytest.mark.parametrize("status", [None, "finished", "unknown", "Completed"])
def test_missing_or_unrecognized_status_is_flagged_not_silent(tmp_path: Path, status: str | None) -> None:
    _standard_run(tmp_path, status=status)
    assert set(_actions(tmp_path).values()) == {("keep", "UNKNOWN_STATUS")}


def test_keep_marker_blocks_everything(tmp_path: Path) -> None:
    run = _standard_run(tmp_path)
    (run / "KEEP").touch()
    assert set(_actions(tmp_path).values()) == {("keep", "PINNED_OR_KEEP")}


def _lock(run_id: str | None, *, corrupt: bool = False) -> LockStatus:
    return LockStatus(who="x", job="j", run_id=run_id, host_pid=1, started_at="t", alive=True, corrupt=corrupt)


def test_run_holding_gpu_lock_is_blocked_others_are_not(tmp_path: Path) -> None:
    _standard_run(tmp_path, "busy")
    _standard_run(tmp_path, "idle")
    p = plan(tmp_path, now=NOW, lock_reader=lambda _p: _lock("busy"))
    blocked = {r.run_id: {i.blocked_by for i in r.items} for r in p.runs}
    assert blocked == {"busy": {"GPU_LOCK_HOLDER"}, "idle": {None}}


def test_corrupt_lock_blocks_all_runs(tmp_path: Path) -> None:
    _standard_run(tmp_path, "a")
    _standard_run(tmp_path, "b")
    p = plan(tmp_path, now=NOW, lock_reader=lambda _p: _lock(None, corrupt=True))
    assert p.actionable == []


def test_plan_is_read_only(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    plan(tmp_path, now=NOW, lock_reader=no_lock)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_apply_deletes_strips_logs_and_keeps_whitelist(tmp_path: Path) -> None:
    run = _standard_run(tmp_path)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    res = apply(tmp_path, [i.item_id for i in p.actionable], by="tester", now=NOW, lock_reader=no_lock)
    assert {i.path.split("/")[-1] for i in res.done} == {
        "epoch_0_step_10",
        "epoch_0_step_20",
        "epoch_0_step_30",
        "epoch_0_step_50",
    }
    ck = run / "checkpoints"
    assert not (ck / "epoch_0_step_10").exists()
    assert not (ck / "epoch_0_step_50").exists()
    for stripped in ("epoch_0_step_20", "epoch_0_step_30"):
        assert (ck / stripped / "model" / "adapter_model.safetensors").exists()
        assert not (ck / stripped / "optim").exists()
        assert not (ck / stripped / "rng").exists()
    assert (ck / "epoch_0_step_40" / "optim").exists()
    assert (ck / "training.jsonl").exists()
    assert (run / "run.json").exists()
    assert (ck / "LATEST").is_symlink()
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert {e["event"] for e in events} == {"checkpoint_pruned"}
    assert {e["by"] for e in events} == {"tester"}
    assert (tmp_path / "outputs" / "system" / "cleanup.jsonl").exists()


def test_apply_skips_items_whose_state_changed(tmp_path: Path) -> None:
    run = _standard_run(tmp_path)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    (run / "KEEP").touch()  # user pins the run after looking at the plan
    res = apply(tmp_path, [i.item_id for i in p.actionable], by="t", now=NOW, lock_reader=no_lock)
    assert res.done == []
    assert len(res.skipped) == len(p.actionable)
    assert (run / "checkpoints" / "epoch_0_step_10").exists()


def test_apply_ignores_unknown_and_whitelisted_ids(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    res = apply(
        tmp_path,
        ["r1/run.json", "r1/checkpoints/training.jsonl", "../../etc", "r1/checkpoints/epoch_0_step_40"],
        by="t",
        now=NOW,
        lock_reader=no_lock,
    )
    assert res.done == []
    assert len(res.skipped) == 4


def test_apply_requires_approval_above_threshold(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    strict = RetentionPolicy(approval_threshold_gb=0.0)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock, policy=strict)
    assert p.needs_approval
    ids = [i.item_id for i in p.actionable]
    with pytest.raises(PermissionError, match="approved-by"):
        apply(tmp_path, ids, by="t", now=NOW, lock_reader=no_lock, policy=strict)
    res = apply(tmp_path, ids, by="t", approved_by="bao-pm", now=NOW, lock_reader=no_lock, policy=strict)
    assert len(res.done) == len(ids)


def test_symlinked_run_or_checkpoint_is_never_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _ckpt(outside, "epoch_0_step_1", age_days=30)
    run = _run(tmp_path, "r1")
    (run / "checkpoints" / "epoch_0_step_9").symlink_to(outside / "checkpoints" / "epoch_0_step_1")
    (tmp_path / "outputs" / "runs" / "linked").symlink_to(outside)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    assert [r.run_id for r in p.runs] == ["r1"]
    assert p.runs[0].items == ()
    assert (outside / "checkpoints" / "epoch_0_step_1").exists()


def test_plan_dict_shape(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    d = plan(tmp_path, now=NOW, lock_reader=no_lock).as_dict()
    assert d["schema_version"] == 1
    assert d["mode"] == "dry_run"
    json.dumps(d)
    run = d["runs"][0]  # type: ignore[index]
    item = run["items"][0]
    assert {"item_id", "path", "roles", "action", "reclaim_gb", "rule", "reason", "blocked_by"} <= item.keys()


def test_splitting_into_small_batches_still_needs_approval(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    first, *rest = sorted(p.actionable, key=lambda i: i.item_id)
    total = sum(i.reclaim_bytes for i in p.actionable)
    policy = RetentionPolicy(approval_threshold_gb=(total - 1) / GIB)  # each batch alone is under the threshold
    apply(tmp_path, [first.item_id], by="t", now=NOW, lock_reader=no_lock, policy=policy)
    with pytest.raises(PermissionError, match="without approval"):
        apply(tmp_path, [i.item_id for i in rest], by="t", now=NOW + 60, lock_reader=no_lock, policy=policy)
    # outside the window the earlier delete no longer counts
    later = NOW + policy.approval_window_s + 120
    res = apply(tmp_path, [i.item_id for i in rest], by="t", now=later, lock_reader=no_lock, policy=policy)
    assert len(res.done) == len(rest)


def test_apply_refuses_wrong_project_root(tmp_path: Path) -> None:
    _standard_run(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "someone-else"\n')
    with pytest.raises(ValueError, match="expected 'vlm-pipeline'"):
        apply(tmp_path, ["r1/checkpoints/epoch_0_step_10"], by="t", now=NOW, lock_reader=no_lock)
    assert (tmp_path / "outputs" / "runs" / "r1" / "checkpoints" / "epoch_0_step_10").exists()
