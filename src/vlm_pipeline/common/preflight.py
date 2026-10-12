"""Pre-run check before taking the GPU lock: memory + swap + disk + lock (ADR-005 D8, Proposal 007 R1-R2).

The ONLY entry point callers use (job runner, ``scripts/cleanup/preflight.py``, graph executor).
Contract: .team/contracts/S-4-preflight.md. Read-only: never removes a lock, never drops caches. Stdlib only.

Each failed condition is a ``Finding`` carrying a catalog ``code`` + ``params`` (rendered through i18n by the app);
``reason`` is an English one-liner for logs and the CLI only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from vlm_pipeline.common.diskguard import DiskSnapshot, LockStatus, check_disk, check_lock, read_disk, required_disk_gb
from vlm_pipeline.common.errors import ExitCode
from vlm_pipeline.common.gpu_lock import default_lock_path
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
_EXIT_PRIORITY = (ExitCode.LOCK_BUSY, ExitCode.NO_MEM, ExitCode.NO_DISK)


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
    available = effective_available_gb(snapshot, cgroup_available_gb)
    required = thresholds.required_available_gb(est_peak_gb)
    findings: list[Finding] = []
    if available < required:
        findings.append(
            Finding(
                "MEM-001",
                ExitCode.NO_MEM,
                f"MemAvailable {available:.1f} GB < required {required:.1f} GB "
                f"(est_peak {est_peak_gb:.1f} GB + margin, hard floor {thresholds.hard_floor_gb:.0f} GB)",
                {"need_gb": round(required, 2), "avail_gb": round(available, 2), "est_peak_gb": est_peak_gb},
            )
        )
    if snapshot.swap_free_gb < thresholds.min_swap_free_gb:
        findings.append(
            Finding(
                "MEM-001",
                ExitCode.NO_MEM,
                f"SwapFree {snapshot.swap_free_gb:.1f} GB < {thresholds.min_swap_free_gb:.0f} GB "
                "(host under memory pressure)",
                {"need_gb": round(required, 2), "avail_gb": round(available, 2), "est_peak_gb": est_peak_gb},
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
) -> Verdict:
    """Run every pre-run check; defaults read the live host so no caller can skip disk or lock by accident."""
    base = root if root is not None else project_root()
    mem = snapshot if snapshot is not None else read_meminfo()
    # The container's own cgroup limit only matters when the caller reads live memory in that container.
    cgroup_gb = read_cgroup_available_gb() if snapshot is None else None
    dsk = disk if disk is not None else read_disk(base)
    if lock_holder is not None:
        lock = lock_holder()
    else:
        lock = check_lock(base / ".team" / "GPU_LOCK" if root is not None else default_lock_path())

    need_disk_gb = required_disk_gb(ckpt_gb)
    disk_params = {"need_gb": round(need_disk_gb, 2), "avail_gb": round(dsk.free_gb, 2)}
    findings = [
        *check_lock_status(lock),
        *check_memory(est_peak_gb, snapshot=mem, cgroup_available_gb=cgroup_gb, thresholds=thresholds),
        *(Finding("DISK-001", ExitCode.NO_DISK, r, disk_params) for r in check_disk(need_disk_gb, dsk)),
    ]
    details: dict[str, object] = {
        "mem": {
            "mem_available_gb": round(effective_available_gb(mem, cgroup_gb), 2),
            "swap_free_gb": round(mem.swap_free_gb, 2),
            "cached_gb": round(mem.cached_gb, 2),
            "est_peak_gb": est_peak_gb,
            "required_gb": round(thresholds.required_available_gb(est_peak_gb), 2),
        },
        "disk": {"path": dsk.path, "free_gb": round(dsk.free_gb, 2), "required_gb": round(need_disk_gb, 2)},
        "lock": None
        if lock is None
        else {"who": lock.who, "job": lock.job, "alive": lock.alive, "corrupt": lock.corrupt},
        "thresholds": {
            "min_margin_gb": thresholds.min_margin_gb,
            "margin_frac": thresholds.margin_frac,
            "hard_floor_gb": thresholds.hard_floor_gb,
            "min_swap_free_gb": thresholds.min_swap_free_gb,
        },
    }
    return combine(findings, details=details)


__all__ = ["Finding", "Verdict", "check_lock_status", "check_memory", "combine", "preflight"]
