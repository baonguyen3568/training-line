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


def test_leases_of_a_dead_executor_are_reaped(tmp_path: Path) -> None:
    from vlm_pipeline.artifacts.stale_runs import reap_resume_leases
    from vlm_pipeline.common.run_claims import LEASES_REL

    leases = tmp_path / LEASES_REL
    leases.mkdir(parents=True)
    for n, verdict in ((2, True), (3, False), (4, None)):
        (leases / f"t-0123456789abcdef-a{n}.json").write_text(json.dumps({"run": "r", "verdict": verdict}))
    (leases / "t-0123456789abcdef-a5.json").write_text("garbage")  # unreadable: left for retention to fail closed
    reaped = reap_resume_leases(tmp_path, gone=lambda body: body["verdict"])  # type: ignore[arg-type,return-value]
    assert reaped == ["t-0123456789abcdef-a2"]  # only "certainly gone"; "cannot tell" is kept
    assert len(list(leases.iterdir())) == 3


def test_holder_identity_beyond_the_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """security L1: another boot -> gone; another PID namespace -> cannot tell; reused pid -> gone."""
    import os

    from vlm_pipeline.common import run_claims

    me = {"pid": os.getpid(), **run_claims.process_identity()}
    assert run_claims.holder_gone(me) is False  # this very process
    assert run_claims.holder_gone({**me, "boot_id": "another-boot"}) is True
    assert run_claims.holder_gone({**me, "pidns": "pid:[1]"}) is None
    assert run_claims.holder_gone({**me, "starttime": "1"}) is True  # same pid, other start: reused
    assert run_claims.holder_gone({"pid": os.getpid()}) is None  # an older lease without the fields
    monkeypatch.setattr(run_claims, "_boot_id", lambda: None)  # /proc hidden here: never fail open
    assert run_claims.holder_gone({**me, "boot_id": "another-boot"}) is None
