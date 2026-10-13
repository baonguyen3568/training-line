"""Unit tests for ``vlm_pipeline.artifacts.retention`` on fake run directories (tmp_path only)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from string import Formatter

import pytest

from vlm_pipeline.artifacts.retention import CONSEQUENCE_CODES, RetentionPolicy, apply, plan
from vlm_pipeline.common.diskguard import LockStatus
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.i18n import load as load_i18n

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


def _run(root: Path, run_id: str, *, status: str | None = "completed", ended_days_ago: float | None = None) -> Path:
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    run = root / "outputs" / "runs" / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "checkpoints" / "training.jsonl").write_text("{}\n")
    if status is not None:
        manifest: dict[str, object] = {"run_id": run_id, "status": status}
        if ended_days_ago is not None:
            manifest["ended_at"] = NOW - ended_days_ago * DAY
        (run / "run.json").write_text(json.dumps(manifest))
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


@pytest.mark.parametrize("status", ["queued", "running", "interrupted", "evicted"])
def test_unfinished_runs_are_never_touched(tmp_path: Path, status: str) -> None:
    _standard_run(tmp_path, status=status)
    assert set(_actions(tmp_path).values()) == {("keep", "RUN_NOT_FINISHED")}


@pytest.mark.parametrize("status", [None, "finished", "killed", "unknown", "Completed"])
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


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize(
    ("ended_days_ago", "expected"),
    [
        (1, {"epoch_0_step_10": "keep", "epoch_0_step_20": "keep", "epoch_0_step_30": "keep"}),  # grace
        (4, {"epoch_0_step_10": "delete", "epoch_0_step_20": "delete", "epoch_0_step_30": "keep"}),  # LATEST waits
        (8, {"epoch_0_step_10": "delete", "epoch_0_step_20": "delete", "epoch_0_step_30": "delete"}),
    ],
)
def test_failed_and_cancelled_runs(
    tmp_path: Path, status: str, ended_days_ago: float, expected: dict[str, str]
) -> None:
    run = _run(tmp_path, "r1", status=status, ended_days_ago=ended_days_ago)
    for name in ("epoch_0_step_10", "epoch_0_step_20", "epoch_0_step_30"):
        _ckpt(run, name, age_days=ended_days_ago, incomplete=name == "epoch_0_step_20")
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_30")
    _age_tree(run / "checkpoints", ended_days_ago)
    got = {k: action for k, (action, _blocked) in _actions(tmp_path).items()}
    assert got == expected


def _age_tree(path: Path, days: float) -> None:
    ts = NOW - days * DAY
    for p in [path, *path.rglob("*")]:
        os.utime(p, (ts, ts), follow_symlinks=False)


def test_backdated_ended_at_never_brings_cleanup_forward(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", status="failed", ended_days_ago=30)
    _ckpt(run, "epoch_0_step_10", age_days=0)
    _age_tree(run / "checkpoints", 30)
    os.utime(run / "checkpoints" / "epoch_0_step_10" / "model", (NOW - 3600, NOW - 3600))  # written 1 h ago
    assert _actions(tmp_path) == {"epoch_0_step_10": ("keep", None)}


def test_failed_run_without_ended_at_uses_run_json_mtime(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", status="failed")
    _ckpt(run, "epoch_0_step_10", age_days=0)
    ts = NOW - 1 * DAY
    os.utime(run / "run.json", (ts, ts))
    assert _actions(tmp_path) == {"epoch_0_step_10": ("keep", None)}  # still in the 3-day grace


def test_touching_latest_carries_the_right_consequence(tmp_path: Path) -> None:
    _standard_run(tmp_path)  # LATEST=step_30 (strip), LOWEST_VAL=step_20 (strip), step_10 (delete)
    run = _run(tmp_path, "f", status="cancelled", ended_days_ago=8)
    _ckpt(run, "epoch_0_step_1", age_days=8)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_1")
    _age_tree(run / "checkpoints", 8)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    got = {
        i.item_id: (i.action, i.consequence_code, dict(i.consequence_params))
        for r in p.runs
        for i in r.items
        if i.action != "keep"
    }
    assert got == {
        "f/checkpoints/epoch_0_step_1": ("delete", "PERMANENT_LOSS", {"name": "f/epoch_0_step_1"}),
        "r1/checkpoints/epoch_0_step_10": ("delete", None, {}),
        "r1/checkpoints/epoch_0_step_20": ("strip_optim", None, {}),
        "r1/checkpoints/epoch_0_step_30": ("strip_optim", "RESUME_IMPOSSIBLE", {"run": "r1"}),
        "r1/checkpoints/epoch_0_step_50": ("delete", None, {}),
    }


@pytest.mark.parametrize("locale", ["vi", "en"])
def test_every_consequence_code_has_text_and_params_cover_placeholders(locale: str) -> None:
    texts = load_i18n(locale)
    for code, params in CONSEQUENCE_CODES.items():
        key = f"storage.consequence.{code}"
        assert key in texts, key
        placeholders = {name for _, name, _, _ in Formatter().parse(texts[key]) if name}
        assert placeholders <= params, (key, placeholders - params)


def test_keep_inside_one_checkpoint_pins_only_that_checkpoint(tmp_path: Path) -> None:
    run = _standard_run(tmp_path)  # step_10 would be deleted (old, not LATEST/best)
    (run / "checkpoints" / "epoch_0_step_10" / "KEEP").touch()
    got = _actions(tmp_path)
    assert got["epoch_0_step_10"] == ("keep", "PINNED_OR_KEEP")
    assert got["epoch_0_step_50"] == ("delete", None)  # the rest of the run is still cleaned


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_checkpoint_keep_beats_incomplete_and_failed_rules(tmp_path: Path, status: str) -> None:
    run = _run(tmp_path, "r1", status=status, ended_days_ago=30)
    ckpt = _ckpt(run, "epoch_0_step_1", age_days=30, incomplete=True)
    (ckpt / "KEEP").touch()
    _age_tree(run / "checkpoints", 30)
    assert _actions(tmp_path) == {"epoch_0_step_1": ("keep", "PINNED_OR_KEEP")}


def _set_aside(run: Path, name: str, *, age_days: float) -> Path:
    """A ``<name>.not-resumable-<stamp>`` entry as the worker leaves it (stamp and mtimes ``age_days`` old)."""
    ts = NOW - age_days * DAY
    entry = run / f"{name}.not-resumable-{time.strftime('%Y%m%d-%H%M%S', time.localtime(ts))}"
    if name == "checkpoints":
        (entry / "epoch_0_step_99" / "optim").mkdir(parents=True)
        (entry / "epoch_0_step_99" / "optim" / "__0_0.distcp").write_bytes(b"o" * 3000)
    else:
        entry.write_text('{"step": 1}\n')
    _age_tree(entry, age_days)
    return entry


def _graph_node(
    root: Path, name: str = "n2-ft-abc", *, status: str = "completed", ended_days_ago: float | None = 0.5
) -> Path:
    """A node run dir whose current training is ``status`` (ended ``ended_days_ago``), with its own checkpoints/."""
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    node = root / "outputs" / "graph" / "runs" / name
    (node / "checkpoints" / "epoch_0_step_9").mkdir(parents=True)
    (node / ".not-resumable").touch()
    manifest: dict[str, object] = {"run_id": name, "status": status, "stage": "train"}
    if ended_days_ago is not None:
        manifest["ended_at"] = NOW - ended_days_ago * DAY
    (node / "run.json").write_text(json.dumps(manifest))
    return node


def _items(root: Path, **kw: object) -> dict[str, tuple[str, str | None, str | None]]:
    p = plan(root, now=NOW, lock_reader=no_lock, **kw)  # type: ignore[arg-type]
    return {i.path: (i.action, i.blocked_by, i.consequence_code) for r in p.runs for i in r.items}


def test_set_aside_expires_after_three_days_once_a_newer_training_completed(tmp_path: Path) -> None:
    node = _graph_node(tmp_path)  # retrained and completed half a day ago
    old = _set_aside(node, "checkpoints", age_days=4)
    old_metrics = _set_aside(node, "metrics.jsonl", age_days=4)
    young = _set_aside(node, "checkpoints", age_days=1)
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    got = {i.item_id: (i.action, i.rule, i.consequence_code) for r in p.runs for i in r.items}
    assert got == {
        f"graph:n2-ft-abc/{old.name}": ("delete", "D3.set_aside", "PERMANENT_LOSS"),
        f"graph:n2-ft-abc/{old_metrics.name}": ("delete", "D3.set_aside", None),
        f"graph:n2-ft-abc/{young.name}": ("keep", "D3.set_aside_wait", None),
    }  # the node's own checkpoints/ belong to the node cache, not to D3

    result = apply(tmp_path, [i.item_id for i in p.actionable], by="t", now=NOW, lock_reader=no_lock)
    assert len(result.done) == 2
    assert (old.exists(), old_metrics.exists(), young.exists()) == (False, False, True)
    assert (node / ".not-resumable").exists()
    assert (node / "checkpoints" / "epoch_0_step_9").is_dir()
    event = json.loads((node / "events.jsonl").read_text().splitlines()[0])
    assert (event["event"], event["rule"]) == ("checkpoint_pruned", "D3.set_aside")


@pytest.mark.parametrize(
    ("status", "ended_days_ago", "own_checkpoints"),
    [
        ("running", None, True),  # retraining still going
        ("failed", 0.5, True),  # retraining failed: the set-aside copy may be the only weights
        ("cancelled", 0.5, True),
        ("evicted", 0.5, False),  # the newer weights were evicted from the node cache
        ("completed", 0.5, False),  # completed but its checkpoints are gone
        ("completed", 6, True),  # the completed training is older than the set-aside (stamp 5 days ago)
    ],
)
def test_set_aside_without_a_newer_completed_training_is_never_auto_deleted(
    tmp_path: Path, status: str, ended_days_ago: float | None, own_checkpoints: bool
) -> None:
    node = _graph_node(tmp_path, status=status, ended_days_ago=ended_days_ago)
    if not own_checkpoints:
        (node / "checkpoints" / "epoch_0_step_9").rmdir()
        (node / "checkpoints").rmdir()
    entry = _set_aside(node, "checkpoints", age_days=5)
    assert _items(tmp_path)[entry.name] == ("delete", "TRAINED_MODEL", "PERMANENT_LOSS")
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    assert p.actionable == []
    assert p.total_reclaim_bytes == 0


def test_only_copy_set_aside_is_deleted_only_with_typed_confirmation(tmp_path: Path) -> None:
    node = _graph_node(tmp_path, status="failed")
    entry = _set_aside(node, "checkpoints", age_days=5)
    item_id = f"graph:n2-ft-abc/{entry.name}"
    result = apply(tmp_path, [item_id], by="t", now=NOW, lock_reader=no_lock)
    assert result.done == []
    assert entry.exists()
    result = apply(tmp_path, [item_id], by="t", now=NOW, lock_reader=no_lock, confirmed={item_id})
    assert [i.item_id for i in result.done] == [item_id]
    assert not entry.exists()


def test_set_aside_in_outputs_runs_and_its_blockers(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", status="completed", ended_days_ago=0)
    entry = _set_aside(run, "checkpoints", age_days=5)
    assert _items(tmp_path)[entry.name] == ("delete", None, "PERMANENT_LOSS")
    p = plan(tmp_path, now=NOW, lock_reader=lambda _p: _lock(None, corrupt=True))
    assert [(i.path, i.blocked_by) for r in p.runs for i in r.items] == [(entry.name, "GPU_LOCK_HOLDER")]
    (run / "KEEP").touch()
    assert _items(tmp_path)[entry.name] == ("keep", "PINNED_OR_KEEP", None)
    item_id = f"r1/{entry.name}"
    result = apply(tmp_path, [item_id], by="t", now=NOW, lock_reader=no_lock, confirmed={item_id})
    assert result.done == []  # confirmation never overrides KEEP


def test_set_aside_stamp_newer_than_mtime_wins(tmp_path: Path) -> None:
    node = _graph_node(tmp_path, ended_days_ago=0.1)
    entry = node / f"checkpoints.not-resumable-{time.strftime('%Y%m%d-%H%M%S', time.localtime(NOW - DAY))}"
    entry.mkdir()
    _age_tree(entry, 30)  # rename keeps old mtimes; the stamp says it was set aside yesterday
    assert _items(tmp_path)[entry.name] == ("keep", None, None)


@pytest.mark.parametrize(
    "name",
    [
        ".not-resumable",
        "checkpoints.not-resumable-garbage",
        "model.not-resumable-20200101-000000",
        "checkpoints",
        "checkpoints.not-resumable-20200101-000000\n",  # "$" would match before a trailing newline
        "checkpoints.not-resumable-\u0662\u0660\u0662\u0660\u0660\u0661\u0660\u0661-000000",  # Unicode digits
    ],
)
def test_only_known_set_aside_names_are_candidates(tmp_path: Path, name: str) -> None:
    node = _graph_node(tmp_path)
    if not (node / name).exists():  # the .not-resumable marker is already there
        (node / name).mkdir()
    _age_tree(node / name, 30)
    assert plan(tmp_path, now=NOW, lock_reader=no_lock).runs == ()


def test_symlinked_set_aside_is_never_followed(tmp_path: Path) -> None:
    node = _graph_node(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (node / "checkpoints.not-resumable-20200101-000000").symlink_to(outside)
    assert plan(tmp_path, now=NOW, lock_reader=no_lock).runs == ()
