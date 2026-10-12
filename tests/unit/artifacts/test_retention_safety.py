"""Extra safety tests for ``vlm_pipeline.artifacts.retention`` (bao-tester review of a2e00c0). tmp_path only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.retention import CLEANABLE_STATUSES, apply, plan
from vlm_pipeline.common.diskguard import LockStatus

NOW = 1_800_000_000.0
DAY = 86_400.0


def _no_lock(_path: Path) -> LockStatus | None:
    return None


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')


def _old_ckpt(run: Path, name: str, age_days: float = 30) -> Path:
    ckpt = run / "checkpoints" / name
    (ckpt / "model").mkdir(parents=True)
    (ckpt / "model" / "w.safetensors").write_bytes(b"w" * 100)
    ts = NOW - age_days * DAY
    os.utime(ckpt, (ts, ts))
    return ckpt


def _run(root: Path, run_id: str, run_json: str | None) -> Path:
    _project(root)
    run = root / "outputs" / "runs" / run_id
    (run / "checkpoints").mkdir(parents=True)
    if run_json is not None:
        (run / "run.json").write_text(run_json)
    return run


def test_delete_never_follows_symlinks_inside_a_checkpoint(tmp_path: Path) -> None:
    """A checkpoint may link to shared data (e.g. HF cache); deleting the checkpoint removes the link only."""
    precious = tmp_path / "hf_cache" / "model.safetensors"
    precious.parent.mkdir()
    precious.write_bytes(b"shared weights")
    run = _run(tmp_path, "r1", json.dumps({"status": "completed"}))
    ckpt = _old_ckpt(run, "epoch_0_step_1")
    (ckpt / "model" / "linked.safetensors").symlink_to(precious)
    (ckpt / "linked_dir").symlink_to(precious.parent, target_is_directory=True)
    os.utime(ckpt, (NOW - 30 * DAY, NOW - 30 * DAY))
    p = plan(tmp_path, now=NOW, lock_reader=_no_lock)
    res = apply(tmp_path, [i.item_id for i in p.actionable], by="t", now=NOW, lock_reader=_no_lock)
    assert [i.path for i in res.done] == ["checkpoints/epoch_0_step_1"]
    assert not ckpt.exists()
    assert precious.read_bytes() == b"shared weights"


@pytest.mark.parametrize(
    "run_json",
    [None, "{not json", json.dumps(["completed"]), json.dumps({"status": ""}), json.dumps({"status": "running"})],
)
def test_runs_without_a_trustworthy_status_are_never_cleaned(tmp_path: Path, run_json: str | None) -> None:
    """Missing/corrupt/odd run.json (e.g. today's S-3 run has no run.json) => everything kept."""
    run = _run(tmp_path, "r1", run_json)
    _old_ckpt(run, "epoch_0_step_1")
    p = plan(tmp_path, now=NOW, lock_reader=_no_lock)
    assert p.actionable == []


def test_latest_link_pointing_outside_the_run_grants_nothing(tmp_path: Path) -> None:
    other = _run(tmp_path, "other", json.dumps({"status": "running"}))
    foreign = _old_ckpt(other, "epoch_0_step_5")
    run = _run(tmp_path, "r1", json.dumps({"status": "completed"}))
    (run / "checkpoints" / "LATEST").symlink_to(foreign)
    plan_ = plan(tmp_path, now=NOW, lock_reader=_no_lock)
    assert all(i.run_id != "other" for i in plan_.actionable)
    apply(tmp_path, [i.item_id for i in plan_.actionable], by="t", now=NOW, lock_reader=_no_lock)
    assert foreign.exists()


@pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: shared RunStatus enum in common (owner: bao-backend, runner S-2)"
)
def test_cleanable_statuses_belong_to_the_shared_run_status_vocabulary() -> None:
    """Retention keys on run.json ``status``. If the writer uses another word (contract says ``finished``,
    ADR-005/retention say ``completed``) retention silently never cleans anything."""
    from vlm_pipeline.common.runinfo import RunStatus

    assert {s.value for s in RunStatus} >= CLEANABLE_STATUSES
