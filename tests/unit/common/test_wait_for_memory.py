"""wait_for_memory: a job that does not fit the ADR-005 A1 budget waits (no lock held) instead of failing."""

from __future__ import annotations

import threading

import pytest

from vlm_pipeline.common.memguard import MemSnapshot
from vlm_pipeline.common.preflight import MemoryWaitTimeoutError, wait_for_memory


def _feed(*avails: float):  # type: ignore[no-untyped-def]
    it = iter(avails)

    def read() -> MemSnapshot:
        return MemSnapshot(mem_total_gb=121.7, mem_available_gb=next(it), cached_gb=0.0, swap_free_gb=6.0)

    return read


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Tick(threading.Event):
    """A cancel event whose wait() advances the fake clock instead of sleeping."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self.clock = clock

    def wait(self, timeout: float | None = None) -> bool:
        self.clock.t += timeout or 0.0
        return self.is_set()


def test_fits_immediately_without_calling_on_wait() -> None:
    seen: list[dict[str, float]] = []
    assert wait_for_memory(10.0, read=_feed(50.0), read_cgroup=lambda: None, on_wait=seen.append)
    assert seen == []


def test_waits_until_memory_frees_up_and_reports_each_poll() -> None:
    clock = _Clock()
    seen: list[dict[str, float]] = []
    ok = wait_for_memory(20.0, read=_feed(20.0, 22.0, 30.0), read_cgroup=lambda: None, on_wait=seen.append,
                         cancel=_Tick(clock), clock=clock, poll_s=30.0)  # fmt: skip
    assert ok
    assert [w["need_gb"] for w in seen] == [28.0, 28.0]  # 20 + 8 beats 20 / 0.8
    assert [w["avail_gb"] for w in seen] == [20.0, 22.0]
    assert [w["waited_s"] for w in seen] == [0.0, 30.0]


def test_cancel_while_waiting_returns_false() -> None:
    clock = _Clock()
    tick = _Tick(clock)
    tick.set()
    assert wait_for_memory(20.0, read=_feed(10.0), read_cgroup=lambda: None, cancel=tick, clock=clock) is False


def test_timeout_raises_sys_002_with_minutes() -> None:
    clock = _Clock()
    with pytest.raises(MemoryWaitTimeoutError) as ei:
        wait_for_memory(20.0, read=_feed(*[10.0] * 100), read_cgroup=lambda: None, cancel=_Tick(clock),
                        clock=clock, poll_s=60.0, timeout_s=300.0)  # fmt: skip
    assert ei.value.code == "SYS-002"
    assert ei.value.params == {"minutes": 5}
