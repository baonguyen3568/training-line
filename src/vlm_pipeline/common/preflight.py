"""Pre-run check before taking the GPU lock: memory + swap + disk + lock (ADR-005 D8, Proposal 007 R1-R2).

The ONLY entry point callers use (job runner, ``scripts/cleanup/preflight.py``, graph executor).
Contract: .team/contracts/S-4-preflight.md. Read-only: never removes a lock, never drops caches. Stdlib only.

Each failed condition is a ``Finding`` carrying a catalog ``code`` + ``params`` (rendered through i18n by the app);
``reason`` is an English one-liner for logs and the CLI only.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.diskguard import DiskSnapshot, LockStatus, check_disk, check_lock, read_disk, required_disk_gb
from vlm_pipeline.common.errors import ExitCode, VlmPipelineError
from vlm_pipeline.common.gpu_lock import default_lock_path, lock_path_for
from vlm_pipeline.common.gpu_mem import (
    MIN_NATIVE_DRIVER_CUDA_13_3,
    GpuInfo,
    driver_too_old,
    memory_model,
    query_gpus,
    vram_required_gb,
)
from vlm_pipeline.common.memguard import (
    DEFAULT_THRESHOLDS,
    MemSnapshot,
    MemThresholds,
    effective_available_gb,
    read_cgroup_available_gb,
    read_meminfo,
)
from vlm_pipeline.common.paths import project_root

# When several checks fail, the reported exit code follows this order; all findings are still listed.
# A busy lock first: low memory is then usually caused by the job holding the lock.
# A GPU that cannot run the image is permanent: it wins over the transient lock/memory/disk refusals.
_EXIT_PRIORITY = (ExitCode.GPU_ENV, ExitCode.LOCK_BUSY, ExitCode.NO_MEM, ExitCode.NO_DISK)


@dataclass(frozen=True)
class Finding:
    """One failed preflight condition."""

    code: str  # error catalog code, e.g. MEM-001
    exit_code: ExitCode
    reason: str  # English, logs/CLI only
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """Preflight result; ``to_json()`` is printed on stdout by the CLI wrappers and written to events.jsonl."""

    exit_code: ExitCode
    findings: tuple[Finding, ...]
    details: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return self.exit_code is ExitCode.OK

    @property
    def reasons(self) -> list[str]:
        return [f.reason for f in self.findings]

    def to_json(self) -> str:
        return json.dumps(
            {
                **self.details,
                "ok": self.ok,
                "exit_code": int(self.exit_code),
                "reasons": self.reasons,
                "findings": [
                    {"code": f.code, "exit_code": int(f.exit_code), "reason": f.reason, "params": dict(f.params)}
                    for f in self.findings
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def check_memory(
    est_peak_gb: float,
    *,
    snapshot: MemSnapshot,
    cgroup_available_gb: float | None = None,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
) -> list[Finding]:
    """Memory/swap conditions for starting a heavy job. Swap is never counted as available memory."""
    if est_peak_gb < 0:
        raise ValueError(f"est_peak_gb must be >= 0, got {est_peak_gb}")
    # User rule (ADR-005 A1): the job may take at most max_free_frac of the memory free NOW. The host's MemAvailable is
    # the budget; a container memory limit is a second, separate budget checked the same way (the stricter binds).
    available = snapshot.mem_available_gb
    required = thresholds.required_available_gb(est_peak_gb)
    if cgroup_available_gb is not None and cgroup_available_gb < available:
        available = cgroup_available_gb
    max_free_pct = round(100 * thresholds.max_free_frac)
    findings: list[Finding] = []
    if available < required:
        findings.append(
            Finding(
                "MEM-001",
                ExitCode.NO_MEM,
                f"MemAvailable {available:.1f} GB < required {required:.1f} GB "
                f"(est_peak {est_peak_gb:.1f} GB may use at most {max_free_pct}% of the free memory and must leave "
                f">= {thresholds.abort_gb:.0f} GB)",
                {
                    "need_gb": round(required, 2),
                    "avail_gb": round(available, 2),
                    "est_peak_gb": est_peak_gb,
                    "max_free_pct": max_free_pct,
                },
            )
        )
    if snapshot.swap_free_gb < thresholds.min_swap_free_gb:
        findings.append(
            Finding(
                "MEM-001",
                ExitCode.NO_MEM,
                f"SwapFree {snapshot.swap_free_gb:.1f} GB < {thresholds.min_swap_free_gb:.0f} GB "
                "(host under memory pressure)",
                {
                    "need_gb": round(required, 2),
                    "avail_gb": round(available, 2),
                    "est_peak_gb": est_peak_gb,
                    "max_free_pct": max_free_pct,
                },
            )
        )
    return findings


def check_lock_status(status: LockStatus | None) -> list[Finding]:
    """A held, stale or unreadable lock all block the run; this never removes a lock."""
    if status is None:
        return []
    if status.corrupt:
        state = "corrupt"
        reason = "GPU lock file is unreadable: treated as busy, tell the PM (never delete it)"
    elif not status.alive:
        state = "stale"
        reason = f"stale GPU lock (holder PID {status.host_pid} not running): tell the PM, never delete it"
    else:
        state = "busy"
        reason = f"GPU busy: {status.who} is running {status.job!r} since {status.started_at}"
    params = {"who": status.who, "job": status.job, "since": status.started_at, "state": state}
    return [Finding("SYS-001", ExitCode.LOCK_BUSY, reason, params)]


def check_gpu(
    gpus: list[GpuInfo], est_vram_gb: float, *, thresholds: MemThresholds = DEFAULT_THRESHOLDS
) -> list[Finding]:
    """Discrete-GPU checks (D-4, NOT tested on real discrete hardware): free VRAM budget and native driver."""
    if memory_model(gpus) != "discrete":
        return []
    gpu = max(gpus, key=lambda g: g.free_gb or 0.0)  # jobs run on one GPU; the freest one is the best case
    findings: list[Finding] = []
    if driver_too_old(gpu):
        findings.append(
            Finding(
                "GPU-001",
                ExitCode.GPU_ENV,
                f"{gpu.name}: driver {gpu.driver_version} < {MIN_NATIVE_DRIVER_CUDA_13_3} (no forward compatibility)",
                {"driver_version": gpu.driver_version, "min_driver": MIN_NATIVE_DRIVER_CUDA_13_3},
            )
        )
    need = vram_required_gb(est_vram_gb, thresholds.max_free_frac)
    free = gpu.free_gb or 0.0
    if free < need:
        findings.append(
            Finding(
                "MEM-001",
                ExitCode.NO_MEM,
                f"{gpu.name}: free VRAM {free:.1f} GB < required {need:.1f} GB "
                f"(est {est_vram_gb:.1f} GB may use at most {round(100 * thresholds.max_free_frac)}% of the free VRAM)",
                {
                    "need_gb": round(need, 2),
                    "avail_gb": round(free, 2),
                    "est_peak_gb": est_vram_gb,
                    "max_free_pct": round(100 * thresholds.max_free_frac),
                    "memory": "vram",
                },
            )
        )
    return findings


def combine(findings: Iterable[Finding], *, details: Mapping[str, object] | None = None) -> Verdict:
    """Merge findings from all checks; exit code follows ``_EXIT_PRIORITY``, every finding is kept."""
    found = tuple(findings)
    codes = {f.exit_code for f in found}
    exit_code = next((c for c in _EXIT_PRIORITY if c in codes), ExitCode.OK)
    if found and exit_code is ExitCode.OK:
        raise ValueError(f"unknown finding exit codes: {sorted(codes)}")
    return Verdict(exit_code=exit_code, findings=found, details=dict(details or {}))


def preflight(
    est_peak_gb: float,
    *,
    ckpt_gb: float = 0.0,
    snapshot: MemSnapshot | None = None,
    disk: DiskSnapshot | None = None,
    lock_holder: Callable[[], LockStatus | None] | None = None,
    root: Path | None = None,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    gpus: list[GpuInfo] | None = None,
    est_vram_gb: float | None = None,
) -> Verdict:
    """Run every pre-run check; defaults read the live host so no caller can skip disk or lock by accident.

    Discrete GPU (D-4): VRAM is a second budget, ``est_vram_gb`` (default ``est_peak_gb``) <= max_free_frac x free
    VRAM, and a GeForce card needs a native driver for the image's CUDA. UMA (GB10) and CPU-only hosts: host memory
    only, as before. ``gpus=None`` queries nvidia-smi only when the memory is live too (tests pass a snapshot).
    """
    base = root if root is not None else project_root()
    mem = snapshot if snapshot is not None else read_meminfo()
    # The container's own cgroup limit only matters when the caller reads live memory in that container.
    cgroup_gb = read_cgroup_available_gb() if snapshot is None else None
    dsk = disk if disk is not None else read_disk(base)
    if lock_holder is not None:
        lock = lock_holder()
    else:
        lock = check_lock(lock_path_for(base) if root is not None else default_lock_path())

    if gpus is None:
        gpus = query_gpus() if snapshot is None else []
    model = memory_model(gpus)

    need_disk_gb = required_disk_gb(ckpt_gb)
    disk_params = {"need_gb": round(need_disk_gb, 2), "avail_gb": round(dsk.free_gb, 2)}
    findings = [
        *check_lock_status(lock),
        *check_memory(est_peak_gb, snapshot=mem, cgroup_available_gb=cgroup_gb, thresholds=thresholds),
        *(Finding("DISK-001", ExitCode.NO_DISK, r, disk_params) for r in check_disk(need_disk_gb, dsk)),
        *check_gpu(gpus, est_peak_gb if est_vram_gb is None else est_vram_gb, thresholds=thresholds),
    ]
    details: dict[str, object] = {
        "mem": {
            "mem_available_gb": round(effective_available_gb(mem, cgroup_gb), 2),
            "swap_free_gb": round(mem.swap_free_gb, 2),
            "cached_gb": round(mem.cached_gb, 2),
            "est_peak_gb": est_peak_gb,
            "required_gb": round(thresholds.required_available_gb(est_peak_gb), 2),
            "max_free_pct": round(100 * thresholds.max_free_frac),
        },
        "disk": {"path": dsk.path, "free_gb": round(dsk.free_gb, 2), "required_gb": round(need_disk_gb, 2)},
        "lock": None
        if lock is None
        else {"who": lock.who, "job": lock.job, "alive": lock.alive, "corrupt": lock.corrupt},
        "gpu": {
            "memory_model": model,
            "devices": [
                {
                    "name": g.name,
                    "free_gb": g.free_gb,
                    "total_gb": g.total_gb,
                    "cc": list(g.compute_capability),
                    "driver": g.driver_version,
                }
                for g in gpus
            ],
        },
        "thresholds": {
            "max_free_frac": thresholds.max_free_frac,
            "abort_gb": thresholds.abort_gb,
            "min_swap_free_gb": thresholds.min_swap_free_gb,
        },
    }
    return combine(findings, details=details)


class MemoryWaitTimeoutError(VlmPipelineError):
    """The job waited ``minutes`` for free memory (ADR-005 A1 budget) and still did not fit: tell the user."""

    code: ClassVar[str] = "SYS-002"


def wait_for_memory(
    est_peak_gb: float,
    *,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    poll_s: float = 30.0,
    timeout_s: float = 1800.0,
    on_wait: Callable[[dict[str, float]], None] | None = None,
    cancel: threading.Event | None = None,
    read: Callable[[], MemSnapshot] = read_meminfo,
    read_cgroup: Callable[[], float | None] = read_cgroup_available_gb,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Block until a job of ``est_peak_gb`` fits the memory budget (PM 2026-10-13: queue, do not fail).

    Returns True when it fits (immediately if it already does), False when ``cancel`` is set while waiting.
    ``on_wait({need_gb, avail_gb, waited_s})`` is called on every poll that does not fit (the UI shows "waiting
    for free memory"). Past ``timeout_s`` raises ``MemoryWaitTimeoutError`` (SYS-002, params ``minutes``). No lock
    is taken here: callers wait BEFORE acquiring the GPU lock.
    """
    stop = cancel if cancel is not None else threading.Event()
    start = clock()
    need = thresholds.required_available_gb(est_peak_gb)
    while True:
        snapshot = read()
        cgroup = read_cgroup()
        if not check_memory(est_peak_gb, snapshot=snapshot, cgroup_available_gb=cgroup, thresholds=thresholds):
            return True
        waited = clock() - start
        if waited >= timeout_s:
            raise MemoryWaitTimeoutError(
                f"no memory for {est_peak_gb:.1f} GB after {waited / 60:.0f} min",
                params={"minutes": math.ceil(waited / 60)},
            )
        if on_wait is not None:
            avail = snapshot.mem_available_gb if cgroup is None else min(snapshot.mem_available_gb, cgroup)
            on_wait({"need_gb": round(need, 2), "avail_gb": round(avail, 2), "waited_s": waited})
        if stop.wait(poll_s):
            return False


__all__ = [
    "Finding",
    "MemoryWaitTimeoutError",
    "Verdict",
    "check_gpu",
    "check_lock_status",
    "check_memory",
    "combine",
    "preflight",
    "wait_for_memory",
]
