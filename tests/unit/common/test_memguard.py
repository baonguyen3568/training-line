"""Unit tests for vlm_pipeline.common.memguard (CPU only, no real /proc or sleeps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.common.diskguard import DiskSnapshot, LockStatus
from vlm_pipeline.common.errors import ExitCode
from vlm_pipeline.common.memguard import (
    Finding,
    MemGuard,
    MemSnapshot,
    MemThresholds,
    check_memory,
    combine,
    effective_available_gb,
    preflight,
    read_cgroup_available_gb,
    read_meminfo,
)

KB_PER_GB = 1024 * 1024


def _meminfo(path: Path, *, avail_gb: float, swap_free_gb: float = 6.0, cached_gb: float = 20.0) -> Path:
    path.write_text(
        f"MemTotal:       {int(121.7 * KB_PER_GB)} kB\n"
        f"MemFree:        {int(5 * KB_PER_GB)} kB\n"
        f"MemAvailable:   {int(avail_gb * KB_PER_GB)} kB\n"
        f"Cached:         {int(cached_gb * KB_PER_GB)} kB\n"
        f"SwapTotal:      {int(16 * KB_PER_GB)} kB\n"
        f"SwapFree:       {int(swap_free_gb * KB_PER_GB)} kB\n"
        "HugePages_Total:       0\n"
    )
    return path


def _snap(avail: float, swap_free: float = 6.0) -> MemSnapshot:
    return MemSnapshot(mem_total_gb=121.7, mem_available_gb=avail, cached_gb=20.0, swap_free_gb=swap_free)


DISK_OK = DiskSnapshot(path="/x", total_gb=3700.0, free_gb=2000.0)
DISK_FULL = DiskSnapshot(path="/x", total_gb=3700.0, free_gb=10.0)


def _no_lock() -> LockStatus | None:
    return None


def _busy_lock() -> LockStatus | None:
    return LockStatus(who="bao-backend", job="smoke", run_id=None, host_pid=123, started_at="16:00", alive=True)


# --------------------------------------------------------------------------------------------- parsing


def test_read_meminfo_parses_kb_to_gb(tmp_path: Path) -> None:
    snap = read_meminfo(_meminfo(tmp_path / "meminfo", avail_gb=43.25, swap_free_gb=6.5, cached_gb=24.0))
    assert snap.mem_available_gb == pytest.approx(43.25, abs=1e-3)
    assert snap.swap_free_gb == pytest.approx(6.5, abs=1e-3)
    assert snap.cached_gb == pytest.approx(24.0, abs=1e-3)
    assert snap.mem_total_gb == pytest.approx(121.7, abs=1e-3)
    assert snap.swap_total_gb == pytest.approx(16.0, abs=1e-3)


def test_read_meminfo_missing_memavailable_is_error(tmp_path: Path) -> None:
    p = tmp_path / "meminfo"
    p.write_text("MemTotal: 1000 kB\n")
    with pytest.raises(ValueError, match="MemAvailable"):
        read_meminfo(p)


def test_cgroup_unlimited_and_absent(tmp_path: Path) -> None:
    assert read_cgroup_available_gb(tmp_path / "nope") is None
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "memory.current").write_text("123\n")
    assert read_cgroup_available_gb(tmp_path) is None


def test_cgroup_limit_takes_min(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text(f"{30 * 1024**3}\n")
    (tmp_path / "memory.current").write_text(f"{12 * 1024**3}\n")
    cg = read_cgroup_available_gb(tmp_path)
    assert cg == pytest.approx(18.0)
    assert effective_available_gb(_snap(43.0), cg) == pytest.approx(18.0)
    assert effective_available_gb(_snap(10.0), cg) == pytest.approx(10.0)


# --------------------------------------------------------------------------------------- thresholds


@pytest.mark.parametrize(
    ("est_peak", "required"),
    [
        (0.0, 20.0),  # hard floor
        (5.0, 20.0),  # 5 + 10 = 15 -> floor 20
        (16.0, 26.0),  # 16 + max(10, 3.2)
        (40.0, 50.0),  # default est_peak (ADR-005): 40 + max(10, 8)
        (70.0, 84.0),  # 70 + 14 (20 %)
    ],
)
def test_required_available_follows_adr005(est_peak: float, required: float) -> None:
    assert MemThresholds().required_available_gb(est_peak) == pytest.approx(required)


def test_adr005_examples_with_43gb_available() -> None:
    # ADR-005 D8: LoRA 4B (~16 GB) OK, full-FT 8B (~70 GB) refused, at MemAvailable 43 GB.
    assert check_memory(16.0, snapshot=_snap(43.0)) == []
    [f] = check_memory(70.0, snapshot=_snap(43.0))
    assert f.exit_code is ExitCode.NO_MEM
    assert f.detail["need_gb"] == pytest.approx(84.0)


def test_low_swap_refuses_even_with_ram() -> None:
    [f] = check_memory(10.0, snapshot=_snap(100.0, swap_free=1.5))
    assert f.exit_code is ExitCode.NO_MEM
    assert "SwapFree" in f.reason


def test_swap_is_never_counted_as_available() -> None:
    # 15 GB RAM + 15 GB swap free must still refuse a 10 GB job (needs 20 GB RAM).
    assert check_memory(10.0, snapshot=_snap(15.0, swap_free=15.0))


def test_negative_peak_rejected() -> None:
    with pytest.raises(ValueError, match="est_peak_gb"):
        check_memory(-1.0, snapshot=_snap(43.0))


# ------------------------------------------------------------------------------------------ combine


def test_combine_priority_lock_then_mem_then_disk_and_keeps_all_reasons() -> None:
    findings = [
        Finding(ExitCode.NO_DISK, "disk"),
        Finding(ExitCode.NO_MEM, "mem"),
        Finding(ExitCode.LOCK_BUSY, "lock"),
    ]
    v = combine(findings)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert v.reasons == ["disk", "mem", "lock"]
    assert combine(findings[:2]).exit_code is ExitCode.NO_MEM
    assert combine(findings[:1]).exit_code is ExitCode.NO_DISK
    ok = combine([])
    assert ok.ok
    assert ok.exit_code is ExitCode.OK


def test_combine_rejects_unknown_exit_code() -> None:
    with pytest.raises(ValueError, match="unknown"):
        combine([Finding(ExitCode.MEM_PREEMPT, "x")])


# ---------------------------------------------------------------------------------------- preflight


def test_preflight_ok_json_shape() -> None:
    v = preflight(16.0, snapshot=_snap(43.0), disk=DISK_OK, lock_holder=_no_lock)
    assert v.ok
    data = json.loads(v.to_json())
    assert data["ok"] is True
    assert data["exit_code"] == 0
    assert data["reasons"] == []
    assert {"mem", "disk", "lock", "thresholds", "findings"} <= data.keys()
    assert "details" not in data  # one source: details are flattened at top level
    assert data["mem"]["required_gb"] == pytest.approx(26.0)
    assert data["lock"] is None


def test_preflight_reports_everything_with_lock_priority() -> None:
    v = preflight(70.0, ckpt_gb=20.0, snapshot=_snap(43.0), disk=DISK_FULL, lock_holder=_busy_lock)
    assert v.exit_code is ExitCode.LOCK_BUSY
    codes = {f.exit_code for f in v.findings}
    assert codes == {ExitCode.LOCK_BUSY, ExitCode.NO_MEM, ExitCode.NO_DISK}
    assert len(v.reasons) == 3


def test_preflight_disk_needs_headroom_even_without_checkpoint() -> None:
    v = preflight(10.0, ckpt_gb=0.0, snapshot=_snap(43.0), disk=DISK_FULL, lock_holder=_no_lock)
    assert v.exit_code is ExitCode.NO_DISK


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (LockStatus("a", "j", None, 99, None, alive=False), "stale"),
        (LockStatus("?", "?", None, None, None, alive=True, corrupt=True), "unreadable"),
    ],
)
def test_preflight_stale_or_corrupt_lock_is_still_busy(status: LockStatus, needle: str) -> None:
    v = preflight(10.0, snapshot=_snap(43.0), disk=DISK_OK, lock_holder=lambda: status)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert needle in v.reasons[0]
    assert "never delete" in v.reasons[0]


def test_preflight_reads_lock_file_under_root_without_touching_it(tmp_path: Path) -> None:
    lock = tmp_path / ".team" / "GPU_LOCK"
    lock.parent.mkdir()
    lock.write_text("bao-x | job | 2026-10-12T16:00 | PID 1\n")  # legacy text format -> corrupt -> busy
    before = lock.read_bytes()
    v = preflight(10.0, snapshot=_snap(43.0), disk=DISK_OK, root=tmp_path)
    assert v.exit_code is ExitCode.LOCK_BUSY
    assert lock.read_bytes() == before


# ------------------------------------------------------------------------------------------ MemGuard


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _guard(tmp_path: Path, clock: _Clock, exits: list[int]) -> tuple[MemGuard, Path]:
    p = tmp_path / "meminfo"
    g = MemGuard(
        meminfo_path=p,
        cgroup_root=tmp_path / "no-cgroup",
        clock=clock,
        hard_exit=lambda: exits.append(1),
    )
    return g, p


def test_guard_states_and_stop_is_sticky(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=40.0)
    g.poll_once()
    assert g.state == "ok"
    assert not g.should_stop.is_set()

    _meminfo(p, avail_gb=12.0)
    g.poll_once()
    assert g.state == "warn"
    assert not g.should_stop.is_set()

    _meminfo(p, avail_gb=7.5)
    g.poll_once()
    assert g.state == "abort"
    assert g.should_stop.is_set()

    _meminfo(p, avail_gb=40.0)  # recovery does not cancel a stop request
    g.poll_once()
    assert g.should_stop.is_set()
    assert g.min_available_gb == pytest.approx(7.5, abs=1e-3)
    assert exits == []


def test_guard_warn_recovers_to_ok(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=12.0)
    g.poll_once()
    _meminfo(p, avail_gb=30.0)
    g.poll_once()
    assert g.state == "ok"


def test_guard_hard_exit_only_after_grace(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=6.0)
    g.poll_once()
    clock.t = 59.0
    g.poll_once()
    assert exits == []
    clock.t = 61.0
    g.poll_once()
    assert exits == [1]


def test_guard_saving_extends_deadline_to_ceiling(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=6.0)
    g.poll_once()
    with g.saving():
        clock.t = 200.0  # a 9B checkpoint can take minutes on this disk
        g.poll_once()
        assert exits == []
        clock.t = 301.0
        g.poll_once()
        assert exits == [1]


def test_guard_save_done_falls_back_to_normal_grace(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=6.0)
    g.poll_once()
    with g.saving():
        clock.t = 100.0
        g.poll_once()
    g.poll_once()
    assert exits == [1]


def test_guard_emergency_floor_overrides_saving(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g, p = _guard(tmp_path, clock, exits)
    _meminfo(p, avail_gb=6.0)
    g.poll_once()
    with g.saving():
        _meminfo(p, avail_gb=3.5)
        clock.t = 1.0
        g.poll_once()
    assert exits == [1]


def test_guard_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="emergency_gb < abort_gb < warn_gb"):
        MemGuard(thresholds=MemThresholds(warn_gb=8.0, abort_gb=8.0))
    with pytest.raises(ValueError, match="emergency_gb"):
        MemGuard(thresholds=MemThresholds(emergency_gb=9.0))


def test_guard_context_manager_starts_and_stops_thread(tmp_path: Path) -> None:
    p = _meminfo(tmp_path / "meminfo", avail_gb=40.0)
    with MemGuard(meminfo_path=p, cgroup_root=tmp_path / "x", interval_s=0.01) as g:
        assert g._thread is not None
        assert g._thread.is_alive()
    assert g._thread is None
