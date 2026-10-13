"""PM 2026-10-13: a run whose process is gone must not show "running" forever."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.stale_runs import QUIET_S, reconcile
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest, write_manifest

NOW = 1_791_900_000.0
DEAD, LIVE_PID = 999_999, 4242


def _run(root: Path, run_id: str, *, status: RunStatus = RunStatus.RUNNING, pid: int | None = DEAD,
         graph: bool = False, age_s: float = 3 * QUIET_S) -> Path:  # fmt: skip
    run_dir = root / "outputs" / ("graph/runs" if graph else "runs") / run_id
    write_manifest(run_dir, RunManifest(run_id=run_id, name=run_id, stage="train", task="sft", status=status, pid=pid))
    for path in (run_dir, *run_dir.iterdir()):
        os.utime(path, (NOW - age_s, NOW - age_s), follow_symlinks=False)
    return run_dir


def _reconcile(
    root: Path, *, containers: set[str] | None = None, lock: tuple[str | None, int | None] = (None, None)
) -> list[str]:
    return reconcile(
        root, now=NOW, alive=lambda pid: pid == LIVE_PID, containers=lambda: containers, lock_reader=lambda: lock
    )


def test_a_dead_process_run_becomes_interrupted_with_a_reason(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r-dead")
    assert _reconcile(tmp_path, containers=set()) == ["r-dead"]
    manifest = read_manifest(run_dir)
    assert (manifest.status, manifest.error_code, manifest.ended_at) == (RunStatus.INTERRUPTED, "SYS-003", NOW)
    event = json.loads((run_dir / "events.jsonl").read_text().splitlines()[-1])
    assert event == {"ts": NOW, "type": "status_reconciled", "from": "running", "status": "interrupted",
                     "reason": "process_gone"}  # fmt: skip
    assert _reconcile(tmp_path, containers=set()) == []  # once


def test_graph_runs_and_quiet_queued_runs_without_pid(tmp_path: Path) -> None:
    _run(tmp_path, "n3-0123456789abcdef", graph=True)
    _run(tmp_path, "probe", pid=None)  # stuck "running" with no pid (the 09:50 vLLM probes)
    assert sorted(_reconcile(tmp_path, containers=set())) == ["n3-0123456789abcdef", "probe"]


@pytest.mark.parametrize(
    ("kwargs", "containers", "lock"),
    [
        ({"pid": LIVE_PID}, None, (None, None)),  # its process is alive
        ({"pid": None, "age_s": 60.0}, None, (None, None)),  # may be waiting for memory / the lock
        ({"pid": None, "status": RunStatus.QUEUED, "age_s": QUIET_S - 60}, None, (None, None)),
        ({}, {"vlm-r1"}, (None, None)),  # its container still runs
        ({}, None, ("r1", LIVE_PID)),  # it is the lock's run and the holder is alive
        ({"status": RunStatus.COMPLETED}, None, (None, None)),  # finished runs are never touched
    ],
)
def test_any_sign_of_life_keeps_the_run(tmp_path: Path, kwargs: dict, containers: set[str] | None,
                                        lock: tuple[str | None, int | None]) -> None:  # fmt: skip
    _run(tmp_path, "r1", **kwargs)
    assert _reconcile(tmp_path, containers=containers, lock=lock) == []


def test_an_unreadable_lock_skips_the_pass(tmp_path: Path) -> None:
    from vlm_pipeline.artifacts import stale_runs

    _run(tmp_path, "r-dead")

    def broken() -> tuple[str | None, int | None]:
        raise stale_runs.LockUnreadableError("malformed")

    assert reconcile(tmp_path, now=NOW, alive=lambda _p: False, containers=set, lock_reader=broken) == []


def test_without_a_container_view_runs_with_a_pid_are_not_judged(tmp_path: Path) -> None:
    """cleanup review: in compose the app sees neither the host PIDs nor the runner's containers."""
    _run(tmp_path, "r-dead")
    _run(tmp_path, "probe", pid=None)  # no pid and quiet for hours: still judged
    assert _reconcile(tmp_path, containers=None) == ["probe"]


def test_writing_a_log_one_level_down_counts_as_activity(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "waiting", pid=None)
    (run_dir / "logs").mkdir()
    os.utime(run_dir / "logs", (NOW - 3 * QUIET_S, NOW - 3 * QUIET_S))
    (run_dir / "logs" / "train.log").write_text("x")
    os.utime(run_dir / "logs" / "train.log", (NOW - 60, NOW - 60))
    assert _reconcile(tmp_path, containers=set()) == []


def test_the_event_is_never_written_through_a_symlink(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "r-dead")
    victim = tmp_path / "victim"
    victim.write_text("")
    (run_dir / "events.jsonl").symlink_to(victim)
    assert _reconcile(tmp_path, containers=set()) == ["r-dead"]  # status fixed, pass not stopped
    assert victim.read_text() == ""
    assert read_manifest(run_dir).status == RunStatus.INTERRUPTED
