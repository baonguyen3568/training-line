"""Job memory budget = 80% of what is FREE at start (user rule, ADR-005 A1 "SỬA LẠI", 2026-10-13; impl. bao-fixcode).

Preflight: est_peak <= max_free_frac * MemAvailable and est_peak + abort_gb <= MemAvailable (a job never starts
already below its abort level), SwapFree >= 2 GB; MEM-001 carries ``max_free_pct``. Runtime: A0 = host MemAvailable at
the guard's first poll, fixed for the job; warn below (1 - warn_free_frac) * A0, controlled stop below
(1 - max_free_frac) * A0, the absolute floors (abort 8 GB, warn 14 GB, emergency 4 GB) win when stricter.
The fraction can only be tightened (MemThresholds raises ValueError, the user path ThresholdTooLooseError CFG-005).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.memguard import MemGuard, MemSnapshot, MemThresholds
from vlm_pipeline.common.preflight import check_memory

TOTAL = 121.7
KB_PER_GB = 1024 * 1024


def _snap(avail: float, swap_free: float = 6.0) -> MemSnapshot:
    return MemSnapshot(mem_total_gb=TOTAL, mem_available_gb=avail, cached_gb=20.0, swap_free_gb=swap_free)


def _meminfo(path: Path, avail_gb: float) -> Path:
    path.write_text(
        f"MemTotal:       {int(TOTAL * KB_PER_GB)} kB\n"
        f"MemFree:        {int(5 * KB_PER_GB)} kB\n"
        f"MemAvailable:   {int(avail_gb * KB_PER_GB)} kB\n"
        f"Cached:         {int(20 * KB_PER_GB)} kB\n"
        f"SwapTotal:      {int(16 * KB_PER_GB)} kB\n"
        f"SwapFree:       {int(6 * KB_PER_GB)} kB\n"
    )
    return path


# ------------------------------------------------------------------------------------------ setting


def test_defaults_are_80_and_75_percent_of_free_memory() -> None:
    t = MemThresholds()
    assert (t.max_free_frac, t.warn_free_frac) == (0.80, 0.75)


@pytest.mark.parametrize("frac", [0.9, 0.81, 1.0, 0.0, -0.1])
def test_a_looser_or_invalid_budget_is_refused_not_clamped(frac: float) -> None:
    with pytest.raises(ValueError, match="max_free_frac"):
        MemThresholds(max_free_frac=frac)


def test_a_tighter_budget_is_accepted_and_warn_stays_below_it() -> None:
    assert MemThresholds(max_free_frac=0.7, warn_free_frac=0.65).max_free_frac == 0.7
    with pytest.raises(ValueError, match="warn_free_frac"):
        MemThresholds(max_free_frac=0.7, warn_free_frac=0.75)


@pytest.mark.parametrize("frac", [0.9, 0.05])
def test_user_override_path_refuses_out_of_range(frac: float) -> None:
    from vlm_pipeline.common.jobrun import ThresholdTooLooseError, tightened_thresholds

    with pytest.raises(ThresholdTooLooseError) as exc:
        tightened_thresholds(max_free_frac=frac)
    assert exc.value.code == "CFG-005"
    assert exc.value.params.get("param") == "max_free_frac"


def test_user_override_path_accepts_tighter() -> None:
    from vlm_pipeline.common.jobrun import tightened_thresholds

    t = tightened_thresholds(max_free_frac=0.7)
    assert t.max_free_frac == 0.7
    assert t.warn_free_frac < t.max_free_frac


# ------------------------------------------------------------------------------------------ preflight


def test_preflight_allows_exactly_80_percent_of_free() -> None:
    assert check_memory(40.0, snapshot=_snap(50.0)) == []


def test_preflight_refuses_above_80_percent_of_free() -> None:
    [finding] = check_memory(40.1, snapshot=_snap(50.0))
    assert finding.code == "MEM-001"
    assert finding.params["max_free_pct"] == 80
    assert finding.params["est_peak_gb"] == 40.1


def test_preflight_never_starts_a_job_below_its_abort_floor() -> None:
    # 22 GB is within 80% of 30 GB, but 22 + 8 (abort floor) = 30 exactly: allowed; 22.5 would start under the floor.
    assert check_memory(22.0, snapshot=_snap(30.0)) == []
    assert check_memory(22.5, snapshot=_snap(30.0)) != []


def test_preflight_budget_follows_what_is_free_not_the_device_total() -> None:
    # Same 121.7 GB device: 100 GB free allows an 80 GB job, 20 GB free allows almost nothing.
    assert check_memory(80.0, snapshot=_snap(100.0)) == []
    assert check_memory(10.0, snapshot=_snap(15.0)) != []


def test_preflight_still_requires_swap_headroom() -> None:
    assert check_memory(1.0, snapshot=_snap(100.0, swap_free=1.0)) != []


def test_a_tighter_budget_is_applied_by_preflight() -> None:
    tight = MemThresholds(max_free_frac=0.5, warn_free_frac=0.45)
    assert check_memory(30.0, snapshot=_snap(50.0), thresholds=tight) != []
    assert check_memory(25.0, snapshot=_snap(50.0), thresholds=tight) == []


# --------------------------------------------------------------------------------- resolved thresholds


@pytest.mark.parametrize(("a0", "abort", "warn"), [(50.0, 10.0, 14.0), (100.0, 20.0, 25.0), (30.0, 8.0, 14.0)])
def test_levels_relative_to_start_with_absolute_floors(a0: float, abort: float, warn: float) -> None:
    t = MemThresholds().for_start(a0)
    assert t.abort_gb == pytest.approx(abort)  # max(8, 0.2*A0)
    assert t.warn_gb == pytest.approx(warn)  # max(14, 0.25*A0)
    assert t.emergency_gb == MemThresholds().emergency_gb


# ------------------------------------------------------------------------------------------- watchdog


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _guard(tmp_path: Path, exits: list[int]) -> tuple[MemGuard, Path]:
    path = tmp_path / "meminfo"
    guard = MemGuard(meminfo_path=path, cgroup_root=None, clock=_Clock(), hard_exit=lambda: exits.append(1))
    return guard, path


def _state(guard: MemGuard) -> str:
    return str(guard.state)  # read fresh each poll (keeps mypy from narrowing a mutable attribute)


def test_watchdog_levels_are_fixed_at_the_first_poll(tmp_path: Path) -> None:
    exits: list[int] = []
    guard, path = _guard(tmp_path, exits)
    _meminfo(path, 100.0)  # A0 = 100: warn < 25, stop < 20
    guard.poll_once()
    assert _state(guard) == "ok"
    _meminfo(path, 26.0)
    guard.poll_once()
    assert _state(guard) == "ok"  # A0 is not re-read: 26 is not "the new start"
    _meminfo(path, 24.0)
    guard.poll_once()
    assert _state(guard) == "warn"
    assert not guard.should_stop.is_set()
    _meminfo(path, 20.5)
    guard.poll_once()
    assert not guard.should_stop.is_set()
    _meminfo(path, 19.5)  # spent more than 80% of what was free at start: controlled stop, no hard kill yet
    guard.poll_once()
    assert _state(guard) == "abort"
    assert guard.should_stop.is_set()
    assert exits == []


def test_small_start_uses_the_absolute_floor(tmp_path: Path) -> None:
    exits: list[int] = []
    guard, path = _guard(tmp_path, exits)
    _meminfo(path, 30.0)  # 20% of 30 = 6 < 8: the 8 GB floor decides
    guard.poll_once()
    _meminfo(path, 8.5)
    guard.poll_once()
    assert not guard.should_stop.is_set()
    _meminfo(path, 7.5)
    guard.poll_once()
    assert guard.should_stop.is_set()


def test_emergency_floor_still_kills_at_once(tmp_path: Path) -> None:
    exits: list[int] = []
    guard, path = _guard(tmp_path, exits)
    _meminfo(path, 50.0)
    guard.poll_once()
    _meminfo(path, 3.5)
    guard.poll_once()
    assert exits == [1]
