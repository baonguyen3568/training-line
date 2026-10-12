"""Unified-memory guard for the shared DGX Spark (GB10, 128 GB UMA).

CPU, GPU, page cache and every other service on the host share one memory pool, so ``nvidia-smi`` reports no
memory and ``cudaMemGetInfo`` under-reports. The only reliable signal is ``MemAvailable`` in ``/proc/meminfo``
(also host-wide inside a container without lxcfs). Under severe pressure the host can freeze instead of raising
CUDA OOM, so we refuse to start (``common.preflight``) and stop in a controlled way (``MemGuard``) well before.

Thresholds: ADR-005 D8 / Proposal 007 R2-R3. Stdlib only, host-safe.
Units: every ``*_gb`` value is GiB (1024**3 bytes), like ``free -g``; ``df -H`` prints powers of 1000.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vlm_pipeline.common.errors import ExitCode
from vlm_pipeline.common.paths import GIB, KIB_PER_GIB

logger = logging.getLogger(__name__)

GuardState = Literal["ok", "warn", "abort"]


@dataclass(frozen=True)
class MemThresholds:
    """Memory thresholds in GiB (ADR-005 D8). Recalibrate from outputs/system/meminfo-*.jsonl."""

    min_margin_gb: float = 10.0
    margin_frac: float = 0.2
    hard_floor_gb: float = 20.0
    min_swap_free_gb: float = 2.0
    warn_gb: float = 14.0
    abort_gb: float = 8.0
    # After a stop request the trainer gets `abort_grace_s` to checkpoint and exit; while it reports a save in
    # progress (MemGuard.saving()) the deadline extends to `saving_grace_s`, unless memory drops below
    # `emergency_gb`, where freezing the shared host is worse than losing an unfinished checkpoint.
    abort_grace_s: float = 60.0
    saving_grace_s: float = 300.0
    emergency_gb: float = 4.0

    def required_available_gb(self, est_peak_gb: float) -> float:
        """MemAvailable needed to start a job with the given estimated peak."""
        margin = max(self.min_margin_gb, self.margin_frac * est_peak_gb)
        return max(est_peak_gb + margin, self.hard_floor_gb)


DEFAULT_THRESHOLDS = MemThresholds()


@dataclass(frozen=True)
class MemSnapshot:
    """Host memory figures in GiB."""

    mem_total_gb: float
    mem_available_gb: float
    cached_gb: float
    swap_free_gb: float
    swap_total_gb: float = 0.0


def read_meminfo(path: Path = Path("/proc/meminfo")) -> MemSnapshot:
    """Parse ``/proc/meminfo`` (values in kB) into GiB."""
    fields: dict[str, float] = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            fields[key] = float(parts[0]) / KIB_PER_GIB
    try:
        return MemSnapshot(
            mem_total_gb=fields["MemTotal"],
            mem_available_gb=fields["MemAvailable"],
            cached_gb=fields.get("Cached", 0.0),
            swap_free_gb=fields.get("SwapFree", 0.0),
            swap_total_gb=fields.get("SwapTotal", 0.0),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing field {exc.args[0]}") from exc


def read_cgroup_available_gb(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """Headroom left under a cgroup v2 memory limit, or None when unlimited / not cgroup v2."""
    try:
        limit = (root / "memory.max").read_text().strip()
        current = (root / "memory.current").read_text().strip()
    except OSError:
        return None
    if limit == "max":
        return None
    return (int(limit) - int(current)) / GIB


def effective_available_gb(snapshot: MemSnapshot, cgroup_available_gb: float | None) -> float:
    """The tighter of host MemAvailable and the container's cgroup headroom."""
    if cgroup_available_gb is None:
        return snapshot.mem_available_gb
    return min(snapshot.mem_available_gb, cgroup_available_gb)


def _hard_exit() -> None:
    os._exit(ExitCode.MEM_PREEMPT)


class MemGuard:
    """Background watchdog: warn below ``warn_gb``, request a controlled stop below ``abort_gb``.

    The trainer polls ``should_stop`` every step, saves a checkpoint and exits with ``ExitCode.MEM_PREEMPT``.
    Wrap checkpoint writes in ``with guard.saving():``. If the process is still alive ``abort_grace_s`` after the
    stop request (``saving_grace_s`` while a save is in progress), or memory falls below ``emergency_gb``,
    ``hard_exit`` kills it (only this process, never anything else on the host).
    """

    def __init__(
        self,
        *,
        thresholds: MemThresholds = DEFAULT_THRESHOLDS,
        interval_s: float = 2.0,
        meminfo_path: Path = Path("/proc/meminfo"),
        cgroup_root: Path | None = Path("/sys/fs/cgroup"),
        clock: Callable[[], float] = time.monotonic,
        hard_exit: Callable[[], None] = _hard_exit,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        """Configure the watchdog (it starts with ``start()`` / ``with``).

        ``cgroup_root=None``: host MemAvailable only (host-side watchdog of a container job, where the job's
        CUDA/UMA memory lives outside any cgroup). ``on_abort`` is called once when the stop is requested (e.g.
        ``docker stop -t <grace>``); ``hard_exit`` once when the grace expires or memory is below ``emergency_gb``.
        """
        if not thresholds.emergency_gb < thresholds.abort_gb < thresholds.warn_gb:
            raise ValueError("thresholds must satisfy emergency_gb < abort_gb < warn_gb")
        self.thresholds = thresholds
        self.interval_s = interval_s
        self.should_stop: threading.Event = threading.Event()
        self.state: GuardState = "ok"
        self.min_available_gb: float = math.inf
        self._meminfo_path = meminfo_path
        self._cgroup_root = cgroup_root
        self._clock = clock
        self._hard_exit = hard_exit
        self._on_abort = on_abort
        self.hard_exited = False
        self._stop_requested_at: float | None = None
        self._saving = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> MemSnapshot:
        """Read memory once and update state; safe to call without the background thread (tests)."""
        snapshot = read_meminfo(self._meminfo_path)
        cgroup_gb = read_cgroup_available_gb(self._cgroup_root) if self._cgroup_root is not None else None
        available = effective_available_gb(snapshot, cgroup_gb)
        self.min_available_gb = min(self.min_available_gb, available)
        if available < self.thresholds.abort_gb:
            if not self.should_stop.is_set():
                logger.error(
                    "MemAvailable %.1f GB < abort %.1f GB: requesting controlled stop",
                    available,
                    self.thresholds.abort_gb,
                )
                self._stop_requested_at = self._clock()
                self.should_stop.set()
                if self._on_abort is not None:
                    try:
                        self._on_abort()
                    except Exception:
                        logger.exception("memguard: on_abort failed; hard exit follows after the grace period")
            self.state = "abort"
        elif available < self.thresholds.warn_gb:
            if self.state == "ok":
                logger.warning("MemAvailable %.1f GB < warn %.1f GB", available, self.thresholds.warn_gb)
            if self.state != "abort":
                self.state = "warn"
        elif self.state == "warn":
            self.state = "ok"
        # A stop request is never withdrawn: memory recovering does not make the run safe to continue.
        if not self.hard_exited and self._stop_requested_at is not None and self._deadline_passed(available):
            logger.critical("no exit after stop request (MemAvailable %.1f GB): hard exit", available)
            self.hard_exited = True
            self._hard_exit()
        return snapshot

    def _deadline_passed(self, available_gb: float) -> bool:
        if self._stop_requested_at is None:
            return False
        if available_gb < self.thresholds.emergency_gb:
            return True
        elapsed = self._clock() - self._stop_requested_at
        if self._saving.is_set():
            return elapsed > self.thresholds.saving_grace_s
        return elapsed > self.thresholds.abort_grace_s

    @contextmanager
    def saving(self) -> Iterator[None]:
        """Wrap checkpoint writes so a pending stop does not hard-exit in the middle of a save."""
        self._saving.set()
        try:
            yield
        finally:
            self._saving.clear()

    def _run(self) -> None:
        while not self._halt.wait(self.interval_s):
            try:
                self.poll_once()
            except (OSError, ValueError):
                logger.exception("memguard: cannot read memory; watchdog keeps running")

    def start(self) -> MemGuard:
        if self._thread is not None:
            raise RuntimeError("MemGuard already started")
        self.poll_once()
        self._thread = threading.Thread(target=self._run, name="memguard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
            self._thread = None

    def __enter__(self) -> MemGuard:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


__all__ = [
    "DEFAULT_THRESHOLDS",
    "MemGuard",
    "MemSnapshot",
    "MemThresholds",
    "effective_available_gb",
    "read_cgroup_available_gb",
    "read_meminfo",
]
