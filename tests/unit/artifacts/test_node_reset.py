"""G-4 "Reset this step": every attempt dir of one node goes, the node gets a new generation (P017 claims stay)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.index import ArtifactIndex, OutputRef
from vlm_pipeline.artifacts.node_reset import apply_reset, attempt_dirs, plan_reset
from vlm_pipeline.common.run_claims import node_generation

KEY = "0123456789abcdef"
NODE = f"n1-{KEY}"
KB = 1024


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    return tmp_path


@pytest.fixture
def idx(root: Path) -> ArtifactIndex:
    return ArtifactIndex(root / "outputs" / "graph" / "app.db")


def _attempt(root: Path, name: str, status: str = "failed", *, ckpt: bool = True, **extra: object) -> Path:
    run = root / "outputs" / "graph" / "runs" / name
    (run / "logs").mkdir(parents=True)
    if ckpt:
        (run / "checkpoints" / "epoch_0_step_9").mkdir(parents=True)
        (run / "checkpoints" / "epoch_0_step_9" / "w").write_bytes(b"w" * 4 * KB)
    (run / "run.json").write_text(json.dumps({"run_id": name, "status": status, "ended_at": time.time(), **extra}))
    return run


def _ten_tries(root: Path) -> list[Path]:
    runs = [_attempt(root, NODE)]
    runs += [_attempt(root, f"{NODE}-a{n}", "interrupted", resumed_from={"run": NODE, "checkpoint": "epoch_0_step_9"})
             for n in range(2, 11)]  # fmt: skip
    return runs


def test_plan_lists_every_attempt_of_exactly_this_node(root: Path, idx: ArtifactIndex) -> None:
    runs = _ten_tries(root)
    _attempt(root, f"n1-{'f' * 16}")  # same node id, other config
    _attempt(root, f"n2-{KEY}")
    _attempt(root, f"{NODE}-g1-a2")  # an earlier reset's generation
    plan = plan_reset(root, idx, NODE)
    assert {a.name for a in plan.attempts} == {r.name for r in runs} | {f"{NODE}-g1-a2"}
    assert plan.blocked_by is None  # its own interrupted attempts resuming from each other do not hold it
    assert plan.consequence_code == "PERMANENT_LOSS"
    assert plan.reclaim_bytes >= 11 * 4 * KB


def test_reset_deletes_attempts_bumps_generation_and_keeps_claims(root: Path, idx: ArtifactIndex) -> None:
    runs = _ten_tries(root)
    out = {"m": OutputRef("model", str(runs[0] / "checkpoints"), "h")}
    idx.record(f"k-{KEY}", run_id=NODE, node_type="vlm/train.lora", outputs=out, run_dir=runs[0])
    claims = root / "outputs" / "system" / "runner_claims"
    claims.mkdir(parents=True)
    (claims / NODE).write_text("{}")
    other = _attempt(root, f"n2-{KEY}")
    plan = plan_reset(root, idx, NODE)
    result = apply_reset(root, idx, NODE, by="t", expected=plan)
    assert sorted(result.deleted) == sorted(r.name for r in runs)
    assert attempt_dirs(root, NODE) == []
    assert other.exists()
    assert (claims / NODE).exists()  # a run id is never reused: the new generation gives new names
    assert node_generation(root, NODE) == 1
    assert idx.get(f"k-{KEY}").state == "invalid"  # type: ignore[union-attr]
    log = [json.loads(x) for x in (root / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()]
    assert (log[-1]["event"], log[-1]["generation"], log[-1]["manual"]) == ("node_reset", 1, True)
    _attempt(root, f"{NODE}-g1")
    apply_reset(root, idx, NODE, by="t", expected=plan_reset(root, idx, NODE))
    assert node_generation(root, NODE) == 2


@pytest.mark.parametrize("case", ["running", "lease", "keep", "index_lease", "gpu_lock"])
def test_reset_is_refused_while_an_attempt_may_be_in_use(root: Path, idx: ArtifactIndex, case: str) -> None:
    run = _attempt(root, NODE, "running" if case == "running" else "failed")
    if case == "lease":
        leases = root / "outputs" / "system" / "resume_leases"
        leases.mkdir(parents=True)
        (leases / f"{NODE}-a2.json").write_text(json.dumps({"run": NODE, "checkpoint": "epoch_0_step_9"}))
    if case == "keep":
        (run / "KEEP").touch()
    if case == "index_lease":
        idx.record(f"k-{KEY}", run_id=NODE, node_type="t", outputs={}, run_dir=run)
        idx.lease([f"k-{KEY}"], "graph-run-7")
    lock = root / "lock.json"
    if case == "gpu_lock":
        lock.write_text(json.dumps({"who": "x", "job": "j", "run_id": NODE, "host_pid": 1, "started_at": "t",
                                    "token": "a", "schema_version": 1}))  # fmt: skip
    plan = plan_reset(root, idx, NODE, lock_path=lock)
    assert plan.blocked_by is not None
    with pytest.raises(PermissionError):
        apply_reset(root, idx, NODE, by="t", expected=plan, lock_path=lock)
    assert run.exists()


def test_reset_refuses_when_the_attempts_changed_since_the_dry_run(root: Path, idx: ArtifactIndex) -> None:
    _attempt(root, NODE)
    plan = plan_reset(root, idx, NODE)
    _attempt(root, f"{NODE}-a2")
    with pytest.raises(PermissionError, match="STATE_CHANGED"):
        apply_reset(root, idx, NODE, by="t", expected=plan)


def test_symlinks_and_bad_names_are_never_touched(root: Path, idx: ArtifactIndex) -> None:
    outside = root / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    graph = root / "outputs" / "graph" / "runs"
    graph.mkdir(parents=True)
    (graph / f"{NODE}-a2").symlink_to(outside)
    assert attempt_dirs(root, NODE) == []
    with pytest.raises(ValueError, match="bad_node"):
        plan_reset(root, idx, "../x")
    gens = root / "outputs" / "system" / "node_generations"
    gens.mkdir(parents=True)
    (gens / f"{NODE}.json").symlink_to(outside / "keep.txt")
    assert plan_reset(root, idx, NODE).blocked_by == "GENERATION_UNREADABLE"
    assert (outside / "keep.txt").read_text() == "x"


@pytest.mark.parametrize("store", ["symlink", "file"])
def test_an_unsafe_lease_store_refuses_the_reset(root: Path, idx: ArtifactIndex, store: str) -> None:
    """Security R1: the node's own leases are only known through a plain store; otherwise refuse (fail closed)."""
    run = _attempt(root, NODE)
    system = root / "outputs" / "system"
    system.mkdir(parents=True)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    leases = system / "resume_leases"
    leases.symlink_to(elsewhere) if store == "symlink" else leases.write_text("x")
    assert plan_reset(root, idx, NODE).blocked_by == "RUN_NOT_FINISHED"
    assert run.exists()


def test_an_unreadable_run_json_refuses_the_reset(root: Path, idx: ArtifactIndex) -> None:
    run = _attempt(root, NODE)
    (run / "run.json").write_text("{not json")
    assert plan_reset(root, idx, NODE).blocked_by == "UNKNOWN_STATUS"


def test_a_failed_generation_write_is_still_audited(
    root: Path, idx: ArtifactIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Security R2: every real delete leaves a cleanup.jsonl line, also when a later step fails."""
    from vlm_pipeline.artifacts import node_reset

    _attempt(root, NODE)

    def broken(*_a: object, **_k: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(node_reset, "_write_generation", broken)
    with pytest.raises(OSError, match="No space"):
        apply_reset(root, idx, NODE, by="t", expected=plan_reset(root, idx, NODE))
    (line,) = (root / "outputs" / "system" / "cleanup.jsonl").read_text().splitlines()
    rec = json.loads(line)
    assert (rec["event"], rec["deleted"], rec["error"]) == ("node_reset", [NODE], "OSError: No space left on device")
