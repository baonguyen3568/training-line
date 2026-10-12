"""Disk-space and GPU-lock checks for ``common.preflight`` (ADR-005 D8; .team/contracts/S-4-preflight.md).

Stdlib only and read-only: nothing here writes or deletes files. All I/O is injectable for tests.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.common.gpu_lock import GpuLockError, is_alive, read_lock
from vlm_pipeline.common.paths import GIB


@dataclass(frozen=True)
class DiskSnapshot:
    path: str
    total_gb: float
    free_gb: float


def read_disk(path: Path, *, statvfs: Callable[[str], os.statvfs_result] = os.statvfs) -> DiskSnapshot:
    """Free space available to an unprivileged user (f_bavail) on the filesystem holding ``path``."""
    st = statvfs(str(path))
    return DiskSnapshot(
        path=str(path),
        total_gb=st.f_blocks * st.f_frsize / GIB,
        free_gb=st.f_bavail * st.f_frsize / GIB,
    )


def required_disk_gb(ckpt_gb: float, *, max_recent: int = 2, headroom_gb: float = 50.0) -> float:
    """Space a job needs: the retained checkpoint window plus the one being written, plus headroom for logs/caches."""
    if ckpt_gb < 0 or max_recent < 0 or headroom_gb < 0:
        raise ValueError("ckpt_gb, max_recent and headroom_gb must be non-negative")
    return ckpt_gb * (max_recent + 1) + headroom_gb


def check_disk(required_gb: float, snapshot: DiskSnapshot) -> list[str]:
    """Return failure reasons (empty list = enough space)."""
    if snapshot.free_gb < required_gb:
        return [f"not enough disk on {snapshot.path}: free {snapshot.free_gb:.1f} GB < required {required_gb:.1f} GB"]
    return []


@dataclass(frozen=True)
class LockStatus:
    who: str
    job: str
    run_id: str | None
    host_pid: int | None
    started_at: str | None
    alive: bool
    corrupt: bool = False


def check_lock(lock_path: Path, *, pid_alive: Callable[[int], bool] | None = None) -> LockStatus | None:
    """Read-only view of the GPU lock for preflight, via ``gpu_lock.read_lock``/``is_alive`` (single parser).

    ``None`` means no lock. A dead holder gives ``alive=False``: callers still treat it as busy and must report it
    to the PM; nothing here removes a lock. A malformed/unreadable lock (``GpuLockError``, e.g. not written by
    ``gpu_lock.acquire``) is ``corrupt=True`` and treated as alive.
    """
    try:
        info = read_lock(lock_path)
    except GpuLockError:
        return LockStatus(who="?", job="?", run_id=None, host_pid=None, started_at=None, alive=True, corrupt=True)
    if info is None:
        return None
    return LockStatus(
        who=info.who,
        job=info.job,
        run_id=info.run_id,
        host_pid=info.host_pid,
        started_at=info.started_at,
        alive=is_alive(info) if pid_alive is None else is_alive(info, pid_alive=pid_alive),
    )
