"""Fault injection for the host-side memory watchdog of container jobs (P-1 incident, 2026-10-12).

Uses a real child process and a fake /proc/meminfo: when MemAvailable drops below abort_gb (ADR-005: 8 GiB) the job
must be stopped gracefully, and killed if it ignores the stop. No docker, no GPU.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from vlm_pipeline.common.jobrun import ExitInfo, MemoryPreemptedBeforeStartError, classify_exit, run_guarded
from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS, MemThresholds
from vlm_pipeline.common.runinfo import RunStatus

KB = 1024 * 1024

# A stand-in for a training container: exits 143 when asked to stop (like SIGTERM), unless told to ignore it.
CHILD = """
import pathlib, sys, time
flag, kill, ignore = map(pathlib.Path, sys.argv[1:4])
while True:
    if kill.exists():
        sys.exit(137)
    if flag.exists() and not ignore.exists():
        sys.exit(143)
    time.sleep(0.05)
"""


def _meminfo(path: Path, avail_gb: float) -> None:
    # Atomic like the real /proc/meminfo: the watchdog must never see a half-written file.
    tmp = path.with_suffix(".tmp")
    # 40 GB device: the absolute ADR-005 levels bind (the 80 % rule is tested in test_memguard.py).
    tmp.write_text(f"MemTotal: {int(40 * KB)} kB\nMemAvailable: {int(avail_gb * KB)} kB\nSwapFree: {6 * KB} kB\n")
    tmp.replace(path)


def _setup(tmp_path: Path) -> tuple[Path, list[str], Path, Path]:
    meminfo = tmp_path / "meminfo"
    _meminfo(meminfo, 40.0)
    stop_flag, kill_flag, ignore = tmp_path / "stop", tmp_path / "kill", tmp_path / "ignore"
    argv = [sys.executable, "-c", CHILD, str(stop_flag), str(kill_flag), str(ignore)]
    return meminfo, argv, stop_flag, kill_flag


def _drop_later(meminfo: Path, avail_gb: float, after_s: float) -> threading.Thread:
    t = threading.Timer(after_s, _meminfo, args=(meminfo, avail_gb))
    t.start()
    return t


def test_thresholds_are_locked_to_adr_005() -> None:
    t = DEFAULT_THRESHOLDS
    assert (t.warn_gb, t.abort_gb, t.emergency_gb) == (14.0, 8.0, 4.0)
    assert (t.max_free_frac, t.warn_free_frac, t.min_swap_free_gb) == (0.80, 0.75, 2.0)  # ADR-005 A1


def test_slow_crossing_stops_the_job_gracefully(tmp_path: Path) -> None:
    """RAM drifting slowly below abort_gb: graceful `docker stop` (checkpoint), no kill."""
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    _meminfo(meminfo, 8.1)
    _drop_later(meminfo, 7.95, 0.4)  # 0.15 GB step: far from emergency at this slope
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.05,
    )  # fmt: skip
    assert rc == 143
    assert guard.should_stop.is_set()
    assert not guard.hard_exited
    assert classify_exit(ExitInfo(rc, oom_killed=False, preempted_by_memguard=True)) == (
        RunStatus.INTERRUPTED,
        "MEM-002",
    )


def test_sudden_drop_kills_without_grace(tmp_path: Path) -> None:
    """40 -> 7.5 GB at once (startup spike): no 60 s grace, kill now."""
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    (tmp_path / "ignore").touch()
    _drop_later(meminfo, 7.5, 0.3)
    started = time.monotonic()
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.05,
    )  # fmt: skip
    assert rc == 137
    assert guard.hard_exited
    assert time.monotonic() - started < 5


def test_job_ignoring_the_stop_is_killed_after_grace(tmp_path: Path) -> None:
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    (tmp_path / "ignore").touch()
    _drop_later(meminfo, 7.0, 0.2)
    short = MemThresholds(abort_grace_s=0.5)
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.05, thresholds=short,
    )  # fmt: skip
    assert rc == 137
    assert guard.hard_exited


def test_emergency_floor_kills_immediately(tmp_path: Path) -> None:
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    (tmp_path / "ignore").touch()
    _drop_later(meminfo, 3.0, 0.2)  # straight below emergency_gb: no grace
    started = time.monotonic()
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.05,
    )  # fmt: skip
    assert rc == 137
    assert guard.hard_exited
    assert time.monotonic() - started < 5  # default grace is 60 s: must not wait for it


def test_not_launched_when_memory_is_already_too_low(tmp_path: Path) -> None:
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    _meminfo(meminfo, 7.0)
    with pytest.raises(MemoryPreemptedBeforeStartError):
        run_guarded(argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo)
    assert not (tmp_path / "log").exists()


def test_healthy_job_is_untouched(tmp_path: Path) -> None:
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    threading.Timer(0.3, stop_flag.touch).start()  # the job ends by itself (flag reused as "done")
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=kill_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.05,
    )  # fmt: skip
    assert rc == 143
    assert not guard.should_stop.is_set()
    assert classify_exit(ExitInfo(0, oom_killed=False, preempted_by_memguard=False))[0] == RunStatus.COMPLETED


def test_replay_of_the_p1_incident_curve(tmp_path: Path) -> None:
    """Real MemAvailable at 1 Hz from the P-1 incident (bao-tester), replayed 20x faster: must stop above the floor."""
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "memory" / "p1_parse_v1.1_26.08-2_mem_avail_gb.txt"
    lines = (line.strip() for line in fixture.read_text().splitlines())
    curve = [float(line) for line in lines if line and not line.startswith("#")]
    assert len(curve) == 225
    assert min(curve) < DEFAULT_THRESHOLDS.abort_gb  # the incident did cross the abort threshold
    meminfo, argv, stop_flag, kill_flag = _setup(tmp_path)
    _meminfo(meminfo, curve[0])

    def replay() -> None:
        for gb in curve:
            _meminfo(meminfo, gb)
            time.sleep(0.05)

    threading.Thread(target=replay, daemon=True).start()
    rc, guard = run_guarded(
        argv, tmp_path / "log", on_abort=stop_flag.touch, hard_exit=kill_flag.touch, meminfo_path=meminfo,
        interval_s=0.02,
    )  # fmt: skip
    assert guard.should_stop.is_set()
    assert rc == 137  # fast fall: killed, not a 60 s graceful stop
    assert guard.fast_fall_killed


def test_resolved_levels_are_published_before_the_job_starts(tmp_path: Path) -> None:
    """The E2E gate reads the lock holder's stop level from run.json WHILE it runs (bao-tester): run_guarded hands
    the resolved guard over right after the first poll, before launching anything."""
    meminfo = tmp_path / "meminfo"
    _meminfo(meminfo, 40.0)  # A0 = 40 GB on a 40 GB device: abort = max(8, 0.2 x 40) = 8
    marker = tmp_path / "launched"
    seen: list[tuple[float, bool]] = []
    rc, _ = run_guarded(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        tmp_path / "log",
        on_abort=lambda: None,
        hard_exit=lambda: None,
        meminfo_path=meminfo,
        on_guard_ready=lambda g: seen.append((g.thresholds.abort_gb, marker.exists())),
    )
    assert rc == 0
    assert seen == [(8.0, False)]  # resolved level, and the process had not started yet
