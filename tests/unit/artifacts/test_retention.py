"""Unit tests for ``vlm_pipeline.artifacts.retention`` on fake run directories (tmp_path only)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from string import Formatter

import pytest

from vlm_pipeline.artifacts.cache_retention import RUN_ROOTS
from vlm_pipeline.artifacts.retention import (
    CONSEQUENCE_CODES,
    RetentionPolicy,
    apply,
    plan,
    read_run_status,
    resume_inputs,
)
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


@pytest.mark.parametrize("status", ["queued", "running", "evicted"])  # interrupted: its own 7-day rule (below)
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


def test_runner_claims_survive_every_pass(tmp_path: Path) -> None:
    """ADR-015: ``outputs/system/runner_claims/<run_id>`` outlives its run dir so a run_id is never reused."""
    _standard_run(tmp_path)
    claims = tmp_path / "outputs" / "system" / "runner_claims"
    claims.mkdir(parents=True)
    (claims / "r1").write_text('{"run_id": "r1"}')
    lookalike = claims / "r2" / "checkpoints" / "epoch_0_step_10"  # a claim dir shaped like a run never becomes one
    lookalike.mkdir(parents=True)
    os.utime(lookalike, (NOW - 30 * DAY, NOW - 30 * DAY))
    every = [i.item_id for i in plan(tmp_path, now=NOW, lock_reader=no_lock).actionable]
    assert every
    assert not any("runner_claims" in i or i.startswith("r2/") for i in every)
    forged = ["../system/runner_claims/r1", "../system/runner_claims/r2/checkpoints/epoch_0_step_10"]
    apply(tmp_path, every + forged, by="t", now=NOW, lock_reader=no_lock)
    assert (claims / "r1").read_text() == '{"run_id": "r1"}'
    assert lookalike.is_dir()
    system = Path("outputs") / "system"  # cache eviction only deletes under RUN_ROOTS: none may cover the claims
    assert not any(system.is_relative_to(r) or r.is_relative_to(system) for r in RUN_ROOTS)


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


@pytest.mark.parametrize(
    ("ended_days_ago", "expected"),
    [
        (2, {"epoch_0_step_10": ("keep", "RUN_INTERRUPTED"), "epoch_0_step_30": ("keep", "RUN_INTERRUPTED")}),
        (8, {"epoch_0_step_10": ("delete", None), "epoch_0_step_30": ("delete", None)}),  # then like failed
    ],
)
def test_interrupted_run_is_resumable_for_seven_days_then_like_failed(
    tmp_path: Path, ended_days_ago: float, expected: dict[str, tuple[str, str | None]]
) -> None:
    run = _run(tmp_path, "r1", status="interrupted", ended_days_ago=ended_days_ago)
    for name in ("epoch_0_step_10", "epoch_0_step_30"):
        _ckpt(run, name, age_days=ended_days_ago)
    (run / "checkpoints" / "LATEST").symlink_to("epoch_0_step_30")
    _age_tree(run / "checkpoints", ended_days_ago)
    os.utime(run / "run.json", (NOW - ended_days_ago * DAY, NOW - ended_days_ago * DAY))
    p = plan(tmp_path, now=NOW, lock_reader=no_lock)
    items = {i.path.split("/")[-1]: i for r in p.runs for i in r.items}
    assert {k: (i.action, i.blocked_by) for k, i in items.items()} == expected
    assert items["epoch_0_step_10"].resume_until == NOW + (7 - ended_days_ago) * DAY
    latest = items["epoch_0_step_30"]
    if latest.action == "delete":  # the last checkpoint is never auto-deleted: a person confirms it
        assert latest.consequence_code == "PERMANENT_LOSS"
        assert items["epoch_0_step_10"].consequence_code is None


def test_keep_still_wins_over_an_expired_interrupted_run(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", status="interrupted", ended_days_ago=30)
    _ckpt(run, "epoch_0_step_10", age_days=30)
    _age_tree(run / "checkpoints", 30)
    os.utime(run / "run.json", (NOW - 30 * DAY, NOW - 30 * DAY))
    (run / "KEEP").touch()
    assert _actions(tmp_path) == {"epoch_0_step_10": ("keep", "PINNED_OR_KEEP")}


def _attempt(root: Path, name: str, source: object, *, status: str = "running", graph: bool = False) -> Path:
    run = root / "outputs" / ("graph/runs" if graph else "runs") / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "status": status, "resumed_from": source}))
    return run


def test_source_of_a_running_attempt_is_kept_whole(tmp_path: Path) -> None:
    """P017 C6: every item of the run dir a later attempt resumes from is blocked, also old non-LATEST checkpoints."""
    _standard_run(tmp_path, "src")
    _attempt(tmp_path, "src-a2", {"run": "src", "checkpoint": "epoch_0_step_30"})
    assert set(_actions(tmp_path).values()) == {("keep", "INPUT_OF_RUNNING_ATTEMPT")}
    ids = [f"src/checkpoints/epoch_0_step_{n}" for n in (10, 20, 30, 50)]
    assert apply(tmp_path, ids, by="t", now=NOW, lock_reader=no_lock, confirmed=set(ids)).done == []


@pytest.mark.parametrize(
    ("status", "age_days", "blocked"),
    [
        ("queued", 0, True),
        ("interrupted", 2, True),  # may still resume (ADR-005 A2 window)
        ("interrupted", 8, False),  # window over
        ("completed", 0, False),
        ("failed", 0, False),
    ],
)
def test_which_attempts_keep_their_source(tmp_path: Path, status: str, age_days: float, blocked: bool) -> None:
    _standard_run(tmp_path, "src")
    run = _attempt(tmp_path, "src-a2", {"run": "src", "checkpoint": "epoch_0_step_30"}, status=status)
    os.utime(run / "run.json", (NOW - age_days * DAY, NOW - age_days * DAY))
    assert resume_inputs(tmp_path, now=NOW).blocks(tmp_path / "outputs" / "runs" / "src") is blocked


@pytest.mark.parametrize(
    "source",
    [
        {"run": "../runs/other", "checkpoint": "epoch_0_step_1"},  # a path, not a name
        {"run": "n1-0123456789abcdef", "checkpoint": "../../etc"},
        {"run": "missing-dir", "checkpoint": "epoch_0_step_1"},
        "n1-0123456789abcdef/checkpoints/epoch_0_step_1",  # wrong shape
    ],
)
def test_unreadable_resumed_from_keeps_every_attempt_of_the_node(tmp_path: Path, source: object) -> None:
    graph = tmp_path / "outputs" / "graph" / "runs"
    for name in ("n1-0123456789abcdef", "n1-0123456789abcdef-a2", "n2-0123456789abcdef"):
        (graph / name).mkdir(parents=True)
    _attempt(tmp_path, "n1-0123456789abcdef-a3", source, graph=True)
    inputs = resume_inputs(tmp_path, now=NOW)
    assert inputs.blocks(graph / "n1-0123456789abcdef")
    assert inputs.blocks(graph / "n1-0123456789abcdef-a2")
    assert not inputs.blocks(graph / "n2-0123456789abcdef")  # another node is not affected


@pytest.mark.parametrize("kind", ["fifo", "symlink", "huge"])
def test_a_hostile_run_json_never_hangs_and_fails_closed(tmp_path: Path, kind: str) -> None:
    """A job can write its own run.json: a fifo, a symlink or a huge file is "unreadable" at once (security R1)."""
    run = _run(tmp_path, "r1", status=None)
    path = run / "run.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to("/dev/zero")
    else:
        path.write_text('{"status": "completed", "pad": "' + "x" * (2 << 20) + '"}')
    assert read_run_status(run) == ("unreadable", None)
    graph = tmp_path / "outputs" / "graph" / "runs"
    (graph / "n1-0123456789abcdef").mkdir(parents=True)
    attempt = graph / "n1-0123456789abcdef-a2"
    attempt.mkdir()
    (path.rename(attempt / "run.json") if kind != "symlink" else (attempt / "run.json").symlink_to("/dev/zero"))
    assert resume_inputs(tmp_path, now=NOW).blocks(graph / "n1-0123456789abcdef")  # its source may be in use


KEY = "0123456789abcdef"


@pytest.mark.parametrize(
    ("lease", "blocked_dirs"),
    [
        ({"run": f"n1-{KEY}", "checkpoint": "epoch_0_step_9", "pid": 1}, {f"n1-{KEY}"}),  # valid: only its source
        ({"run": "../x", "checkpoint": "epoch_0_step_9"}, {f"n1-{KEY}", f"n1-{KEY}-a2", f"n1-{KEY}-g2"}),  # whole node
        ("not json", {f"n1-{KEY}", f"n1-{KEY}-a2", f"n1-{KEY}-g2"}),
    ],
)
def test_resume_lease_keeps_the_source_before_run_json_exists(
    tmp_path: Path, lease: object, blocked_dirs: set[str]
) -> None:
    """P017 R2: the executor leases the source while the new attempt waits for memory / the GPU lock."""
    graph = tmp_path / "outputs" / "graph" / "runs"
    names = (f"n1-{KEY}", f"n1-{KEY}-a2", f"n1-{KEY}-g2", f"n1-{'f' * 16}", f"n2-{KEY}")
    for name in names:
        (graph / name).mkdir(parents=True)
    leases = tmp_path / "outputs" / "system" / "resume_leases"
    leases.mkdir(parents=True)
    (leases / f"n1-{KEY}-a3.json").write_text(lease if isinstance(lease, str) else json.dumps(lease))
    (leases / "evil.json").write_text(json.dumps({"run": f"n2-{KEY}", "checkpoint": "epoch_0_step_1"}))  # bad name
    inputs = resume_inputs(tmp_path, now=NOW)
    assert {n for n in names if inputs.blocks(graph / n)} == blocked_dirs  # another node / config is never affected
    assert (leases / f"n1-{KEY}-a3.json").exists()  # read only: leases are released by the executor / stale_runs


@pytest.mark.parametrize("kind", ["symlink", "file", "system_symlink"])
def test_an_unsafe_lease_store_keeps_every_graph_run(tmp_path: Path, kind: str) -> None:
    """Security R2: a job that replaces the lease store must not unprotect the sources (fail closed)."""
    graph = tmp_path / "outputs" / "graph" / "runs"
    (graph / f"n1-{KEY}").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    system = tmp_path / "outputs" / "system"
    if kind == "system_symlink":
        system.symlink_to(elsewhere)
        (elsewhere / "resume_leases").mkdir()
    else:
        system.mkdir()
        leases = system / "resume_leases"
        leases.symlink_to(elsewhere) if kind == "symlink" else leases.write_text("x")
    assert resume_inputs(tmp_path, now=NOW).blocks(graph / f"n1-{KEY}")


def test_set_aside_of_a_resume_source_is_blocked(tmp_path: Path) -> None:
    _run(tmp_path, "pyproject-holder")  # writes pyproject.toml
    node = tmp_path / "outputs" / "graph" / "runs" / "n1-0123456789abcdef"
    aside = node / "checkpoints.not-resumable-20200101-000000"
    (aside / "epoch_0_step_9").mkdir(parents=True)
    _attempt(tmp_path, "n1-0123456789abcdef-a2", {"run": "n1-0123456789abcdef", "checkpoint": "epoch_0_step_9"},
             graph=True)  # fmt: skip
    items = [i for r in plan(tmp_path, now=NOW, lock_reader=no_lock).runs for i in r.items if i.kind == "set_aside"]
    assert [(i.action, i.blocked_by) for i in items] == [("keep", "INPUT_OF_RUNNING_ATTEMPT")]


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


def _eval_copy(root: Path, name: str, *, age_days: float) -> Path:
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    d = root / "data" / "eval" / name
    (d / "images").mkdir(parents=True)
    (d / "images" / "a.png").write_bytes(b"p" * 2000)
    _age_tree(d, age_days)
    return d


def _eval_items(root: Path) -> dict[str, tuple[str, str | None]]:
    p = plan(root, now=NOW, lock_reader=no_lock)
    return {i.run_id: (i.action, i.blocked_by) for r in p.runs for i in r.items if i.kind == "eval_data"}


def test_eval_data_copies_set_aside_or_unfinished(tmp_path: Path) -> None:
    """G-5: invalid copies manual for 7 days, then automatic; crashed installs after 1 day; the dataset in use never."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(NOW - 9 * DAY))
    young_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(NOW - 1 * DAY))
    _eval_copy(tmp_path, "ocrbench-92a54bd1", age_days=30)  # in use
    _eval_copy(tmp_path, f".ocrbench-92a54bd1.invalid-{stamp}-ab12", age_days=9)
    _eval_copy(tmp_path, f".ocrbench-92a54bd1.invalid-{young_stamp}-cd34", age_days=1)
    _eval_copy(tmp_path, ".ocrbench-92a54bd1.0123abcd.tmp", age_days=2)
    _eval_copy(tmp_path, ".ocrbench-92a54bd1.4567abcd.tmp", age_days=0.1)  # maybe installing now: not shown
    _eval_copy(tmp_path, ".ocrbench-92a54bd1.invalid-oops", age_days=30)  # not our name: never touched
    assert _eval_items(tmp_path) == {
        f".ocrbench-92a54bd1.invalid-{stamp}-ab12": ("delete", None),
        f".ocrbench-92a54bd1.invalid-{young_stamp}-cd34": ("delete", "MANUAL_ONLY"),
        ".ocrbench-92a54bd1.0123abcd.tmp": ("delete", None),
    }
    young = f"eval:.ocrbench-92a54bd1.invalid-{young_stamp}-cd34"
    assert apply(tmp_path, [young], by="scheduler", now=NOW, lock_reader=no_lock).done == []  # never automatic
    done = apply(tmp_path, [young], by="person", now=NOW, lock_reader=no_lock, manual=True).done
    assert [i.item_id for i in done] == [young]
    assert (tmp_path / "data" / "eval" / "ocrbench-92a54bd1").is_dir()
    assert not (tmp_path / "data" / "eval" / f".ocrbench-92a54bd1.invalid-{young_stamp}-cd34").exists()
    rec = json.loads((tmp_path / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()[-1])
    assert rec["event"] == "eval_data_pruned"


def test_eval_data_waits_for_a_running_download_and_honors_keep(tmp_path: Path) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(NOW - 9 * DAY))
    kept = _eval_copy(tmp_path, f".ocrbench-92a54bd1.invalid-{stamp}-ab12", age_days=9)
    (kept / "KEEP").touch()
    assert _eval_items(tmp_path) == {kept.name: ("delete", "PINNED_OR_KEEP")}
    active = tmp_path / "outputs" / "system" / "downloads" / "active"
    active.parent.mkdir(parents=True)
    active.touch()
    assert _eval_items(tmp_path) == {}


H24 = "0123456789abcdef01234567"


def _cache_entry(
    root: Path, name: str, *, age_days: float, status: str | None = "completed", file: bool = False
) -> Path:
    if not (root / "pyproject.toml").exists():
        (root / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    p = root / "outputs" / "eval-cache" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if file:
        p.write_text("{}")
    else:
        (p / "logs").mkdir(parents=True)
        (p / "out.jsonl").write_text("{}\n" * 50)
        if status is not None:
            (p / "run.json").write_text(json.dumps({"run_id": name, "status": status}))
    _age_tree(p, age_days)
    return p


def _cache_items(root: Path, **kw: object) -> dict[str, tuple[str, str | None]]:
    p = plan(root, now=NOW, lock_reader=kw.pop("lock_reader", no_lock))  # type: ignore[arg-type]
    return {i.run_id: (i.action, i.blocked_by) for r in p.runs for i in r.items if i.kind == "eval_cache"}


def test_eval_cache_leftovers_go_after_seven_days_the_caches_never(tmp_path: Path) -> None:
    """bao-tester's layout (eval/node_worker.py, eval_cord_worker): only leftovers are items."""
    stale = time.strftime("%Y%m%d-%H%M%S", time.localtime(NOW - 9 * DAY))
    _cache_entry(tmp_path, f"ocrbench-{H24}.json", age_days=90, file=True)  # the cache: never
    _cache_entry(tmp_path, H24, age_days=90)  # the CORD cache: never
    _cache_entry(tmp_path, f"ocrbench-{H24}.run-4242-abc123", age_days=9)
    _cache_entry(tmp_path, f"ocrbench-{H24}.run-4243-abc124", age_days=2)  # too young: not shown
    _cache_entry(tmp_path, f"ocrbench-{H24}.run-4244-abc125", age_days=9, status="running")
    _cache_entry(tmp_path, f"ocrbench-{H24}.run-4245-abc126", age_days=9, status=None)  # never wrote run.json
    _cache_entry(tmp_path, f"{H24}.tmp-77-def456", age_days=9, status="failed")
    _cache_entry(tmp_path, f"{H24}.stale-{stale}", age_days=9, status=None)
    _cache_entry(tmp_path, f".ocrbench-{H24}.json.0123abcd.tmp", age_days=2, file=True)
    _cache_entry(tmp_path, f"ocrbench-{H24}.run-x-abc127", age_days=90)  # not our name
    assert _cache_items(tmp_path) == {
        f"ocrbench-{H24}.run-4242-abc123": ("delete", None),
        f"ocrbench-{H24}.run-4244-abc125": ("delete", "RUN_NOT_FINISHED"),
        f"ocrbench-{H24}.run-4245-abc126": ("delete", None),
        f"{H24}.tmp-77-def456": ("delete", None),
        f"{H24}.stale-{stale}": ("delete", None),
        f".ocrbench-{H24}.json.0123abcd.tmp": ("delete", None),
    }
    done = apply(tmp_path, [f"evalcache:ocrbench-{H24}.run-4242-abc123"], by="t", now=NOW, lock_reader=no_lock).done
    assert len(done) == 1
    assert (tmp_path / "outputs" / "eval-cache" / f"ocrbench-{H24}.json").exists()
    rec = json.loads((tmp_path / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()[-1])
    assert rec["event"] == "eval_cache_pruned"


def test_eval_cache_run_holding_the_gpu_lock_is_kept(tmp_path: Path) -> None:
    name = f"ocrbench-{H24}.run-4242-abc123"
    _cache_entry(tmp_path, name, age_days=9, status="completed")
    held = LockStatus(who="t", job="j", run_id=name, host_pid=1, started_at="s", alive=True)
    assert _cache_items(tmp_path, lock_reader=lambda _p: held) == {name: ("delete", "GPU_LOCK_HOLDER")}


def test_eval_cache_symlinks_fresh_files_and_broken_manifests_are_kept(tmp_path: Path) -> None:
    """bao-tester: a link is never listed/followed; a fresh file deep in the tree keeps it; a broken run.json too."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    _cache_entry(tmp_path, f"ocrbench-{H24}.json", age_days=1, file=True)  # creates the folder
    link = tmp_path / "outputs" / "eval-cache" / f"ocrbench-{H24}.run-1-aaaaaa"
    link.symlink_to(outside)
    fresh = _cache_entry(tmp_path, f"ocrbench-{H24}.run-2-bbbbbb", age_days=9)
    (fresh / "logs" / "stdout.log").write_text("still writing")  # mtime now, the dir itself 9 days old
    os.utime(fresh, (NOW - 9 * DAY, NOW - 9 * DAY))
    os.utime(fresh / "logs" / "stdout.log", (NOW - 3600, NOW - 3600))
    broken = _cache_entry(tmp_path, f"ocrbench-{H24}.run-3-cccccc", age_days=9)
    (broken / "run.json").write_text("{not json")
    _age_tree(broken, 9)
    assert _cache_items(tmp_path) == {broken.name: ("delete", "RUN_NOT_FINISHED")}
    assert (outside / "keep.txt").exists()


def test_eval_cache_names_with_a_catalog_id_prefix(tmp_path: Path) -> None:
    """bao-tester: CORD caches gain the catalog id (cord-v2-<24hex>); old bare-hash leftovers are still cleaned."""
    stale = time.strftime("%Y%m%d-%H%M%S", time.localtime(NOW - 9 * DAY))
    _cache_entry(tmp_path, f"cord-v2-{H24}", age_days=90)  # the cache: never
    _cache_entry(tmp_path, f"cord-v2-{H24}.tmp-77-def456", age_days=9, status="failed")
    _cache_entry(tmp_path, f"cord-v2-{H24}.stale-{stale}", age_days=9, status=None)
    _cache_entry(tmp_path, f"cord-v2-{H24}.run-5-abcdef", age_days=9)
    _cache_entry(tmp_path, f".cord-v2-{H24}.json.0123abcd.tmp", age_days=2, file=True)
    _cache_entry(tmp_path, f"{H24}.stale-{stale}", age_days=9, status=None)  # old bare-hash leftover
    assert set(_cache_items(tmp_path)) == {
        f"cord-v2-{H24}.tmp-77-def456",
        f"cord-v2-{H24}.stale-{stale}",
        f"cord-v2-{H24}.run-5-abcdef",
        f".cord-v2-{H24}.json.0123abcd.tmp",
        f"{H24}.stale-{stale}",
    }
