"""Disk-space and GPU-lock checks for ``memguard.preflight`` (ADR-005 D8; .team/contracts/S-4-preflight.md).

Stdlib only and read-only: nothing here writes or deletes files. All I/O is injectable for tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3


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
        total_gb=st.f_blocks * st.f_frsize / GB,
        free_gb=st.f_bavail * st.f_frsize / GB,
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    return True


def _parse_lock(raw: str) -> dict[str, object]:
    """JSON lock (backend-gpu-job skill / gpu_lock) or legacy one-line text ``who | job | started_at | PID``."""
    try:
        data = json.loads(raw)
    except ValueError:
        parts = [p.strip() for p in raw.strip().split("|")]
        if len(parts) < 4:
            raise
        return {"who": parts[0], "job": parts[1], "started_at": parts[2], "host_pid": parts[3]}
    if not isinstance(data, dict):
        raise ValueError("lock is not a JSON object")
    return data


def check_lock(lock_path: Path, *, pid_alive: Callable[[int], bool] = _pid_alive) -> LockStatus | None:
    """Read-only view of ``.team/GPU_LOCK``. Temporary until ``gpu_lock.read_lock``/``is_alive`` (S-2) exist.

    ``None`` means no lock. A dead holder gives ``alive=False``: callers still treat it as busy and must report it
    to the PM; this function never removes a lock. An unreadable lock is ``corrupt=True`` and treated as alive.
    """
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return LockStatus(who="?", job="?", run_id=None, host_pid=None, started_at=None, alive=True, corrupt=True)
    try:
        data = _parse_lock(raw)
        pid_raw = data.get("host_pid", data.get("pid"))
        host_pid = int(str(pid_raw)) if pid_raw is not None else None
    except (ValueError, TypeError):
        return LockStatus(who="?", job="?", run_id=None, host_pid=None, started_at=None, alive=True, corrupt=True)
    alive = pid_alive(host_pid) if host_pid is not None and host_pid > 0 else True
    return LockStatus(
        who=str(data.get("who", "?")),
        job=str(data.get("job", "?")),
        run_id=str(data["run_id"]) if data.get("run_id") is not None else None,
        host_pid=host_pid,
        started_at=str(data["started_at"]) if data.get("started_at") is not None else None,
        alive=alive,
    )
