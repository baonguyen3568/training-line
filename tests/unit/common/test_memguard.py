"""Unit tests for vlm_pipeline.common.memguard (CPU only, no real /proc or sleeps)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.memguard import (
    MemGuard,
    MemSnapshot,
    MemThresholds,
    effective_available_gb,
    read_cgroup_available_gb,
    read_meminfo,
)

KB_PER_GB = 1024 * 1024


# A 40 GB device: there the % rule (80 % used -> abort below 8 GB, warn below 10 GB) never beats the absolute
# ADR-005 levels (abort 8, warn 14), so these tests pin the absolute layer. The % rule has its own tests below.
SMALL_TOTAL_GB = 40.0


def _meminfo(
    path: Path,
    *,
    avail_gb: float,
    swap_free_gb: float = 6.0,
    cached_gb: float = 20.0,
    total_gb: float = SMALL_TOTAL_GB,
) -> Path:
    path.write_text(
        f"MemTotal:       {int(total_gb * KB_PER_GB)} kB\n"
        f"MemFree:        {int(5 * KB_PER_GB)} kB\n"
        f"MemAvailable:   {int(avail_gb * KB_PER_GB)} kB\n"
        f"Cached:         {int(cached_gb * KB_PER_GB)} kB\n"
        f"SwapTotal:      {int(16 * KB_PER_GB)} kB\n"
        f"SwapFree:       {int(swap_free_gb * KB_PER_GB)} kB\n"
        "HugePages_Total:       0\n"
    )
    return path


def _snap(avail: float, swap_free: float = 6.0) -> MemSnapshot:
    return MemSnapshot(mem_total_gb=SMALL_TOTAL_GB, mem_available_gb=avail, cached_gb=20.0, swap_free_gb=swap_free)


# --------------------------------------------------------------------------------------------- parsing


def test_read_meminfo_parses_kb_to_gb(tmp_path: Path) -> None:
    snap = read_meminfo(
        _meminfo(tmp_path / "meminfo", avail_gb=43.25, swap_free_gb=6.5, cached_gb=24.0, total_gb=121.7)
    )
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
        (0.0, 8.0),  # nothing to run: just not below the abort level
        (16.0, 24.0),  # 16 + 8 beats 16 / 0.8 = 20
        (40.0, 50.0),  # 40 / 0.8 = 50 beats 40 + 8
        (70.0, 87.5),  # 70 / 0.8
    ],
)
def test_required_available_follows_adr005_a1(est_peak: float, required: float) -> None:
    assert MemThresholds().required_available_gb(est_peak) == pytest.approx(required)


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


def _real_p1_curve() -> list[float]:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "memory" / "p1_parse_v1.1_26.08-2_mem_avail_gb.txt"
    return [float(x) for x in (ln.strip() for ln in fixture.read_text().splitlines()) if x and not x.startswith("#")]


def test_p1_curve_at_real_speed_is_killed_before_reaching_abort(tmp_path: Path) -> None:
    """Replay the measured 1 Hz P-1 curve with a fake clock: the fast-fall rule must kill ABOVE abort_gb (8 GB),
    i.e. before the 60 s graceful path would even start, and well above the 6.34 GB floor that was reached."""
    clock, exits = _Clock(), []
    g = MemGuard(meminfo_path=tmp_path / "m", cgroup_root=None, clock=clock, hard_exit=lambda: exits.append(1),
                 interval_s=1.0)  # fmt: skip
    killed_at = None
    for second, gb in enumerate(_real_p1_curve()):
        clock.t = float(second)
        _meminfo(tmp_path / "m", avail_gb=gb)
        g.poll_once()
        if exits:
            killed_at = gb
            break
    assert killed_at is not None
    assert g.fast_fall_killed
    assert killed_at > MemThresholds().abort_gb


def test_slow_drift_below_warn_is_not_killed(tmp_path: Path) -> None:
    clock, exits = _Clock(), []
    g = MemGuard(meminfo_path=tmp_path / "m", cgroup_root=None, clock=clock, hard_exit=lambda: exits.append(1))
    for i in range(60):  # 13.0 -> 10.0 GB over 60 s (0.05 GB/s): needs > 5 s to reach 8 GB
        clock.t = float(i)
        _meminfo(tmp_path / "m", avail_gb=13.0 - 0.05 * i)
        g.poll_once()
    assert exits == []
    assert g.state == "warn"


def test_fast_fall_ignored_while_plenty_of_memory_left(tmp_path: Path) -> None:
    """Weight loading falls fast but plateaus far above the danger zone: no kill above warn_gb."""
    clock, exits = _Clock(), []
    g = MemGuard(meminfo_path=tmp_path / "m", cgroup_root=None, clock=clock, hard_exit=lambda: exits.append(1))
    for i, gb in enumerate([44.0, 40.0, 36.0, 32.0, 28.0, 24.0, 20.0, 18.0, 18.0, 18.0]):
        clock.t = float(i)
        _meminfo(tmp_path / "m", avail_gb=gb)
        g.poll_once()
    assert exits == []


# ------------------------------------------- 80 % of the free memory at start (user rule, ADR-005 A1, 2026-10-13)


def test_job_using_more_than_80_percent_of_start_free_memory_stops(tmp_path: Path) -> None:
    """A0 = 60 GB free at start: warn below 15 GB (75 % of A0 used), stop below 12 GB (80 % used)."""
    path = tmp_path / "meminfo"
    stops: list[int] = []
    ticks = iter(range(0, 10_000, 600))  # polls 10 min apart: a slow drift, not a fast fall
    guard = MemGuard(
        meminfo_path=_meminfo(path, avail_gb=60.0),
        cgroup_root=None,
        clock=lambda: float(next(ticks)),
        hard_exit=lambda: None,
        on_abort=lambda: stops.append(1),
    )
    guard.poll_once()
    assert guard.state == "ok"
    assert guard.thresholds.abort_gb == pytest.approx(12.0)
    assert guard.thresholds.warn_gb == pytest.approx(15.0)
    _meminfo(path, avail_gb=14.0)
    guard.poll_once()
    assert guard.state == "warn"
    assert not stops
    _meminfo(path, avail_gb=11.9)
    guard.poll_once()
    assert guard.state == "abort"
    assert stops == [1]


def test_absolute_floors_win_when_little_was_free_at_start() -> None:
    small = MemThresholds().for_start(30.0)  # 0.2 x 30 = 6 < 8, 0.25 x 30 = 7.5 < 14
    assert (small.abort_gb, small.warn_gb) == (8.0, 14.0)
    assert small.emergency_gb == MemThresholds().emergency_gb  # the absolute last layer never moves
    big = MemThresholds().for_start(100.0)
    assert big.emergency_gb < big.abort_gb < big.warn_gb


@pytest.mark.parametrize("frac", [0.81, 0.9, 1.0, 0.0, -0.1])
def test_max_free_frac_can_only_be_made_stricter(frac: float) -> None:
    with pytest.raises(ValueError, match="max_free_frac"):
        MemThresholds(max_free_frac=frac)
    stricter = MemThresholds(max_free_frac=0.7, warn_free_frac=0.65).for_start(100.0)
    assert stricter.abort_gb == pytest.approx(30.0)
