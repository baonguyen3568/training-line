"""Single-heavy-job GPU lock: ``.team/GPU_LOCK`` (dev host) or the app's lock path (Proposal 008 §2.2).

Stdlib only. The lock is a small JSON document created atomically: it is written to a temporary file first and then
``os.link``-ed into place, so readers never see an empty or half-written lock and creation fails if a lock exists.

Rules (CLAUDE.md "GPU lock"):
- Only the holder releases a lock, proven by its ``token``; nothing here ever deletes someone else's lock.
- A lock whose ``host_pid`` is dead is *stale*. It is reported, never removed automatically (tell the PM).
- ``host_pid`` is the PID of the host process holding the lock for the whole job, never a PID inside a container.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.paths import project_root

LOCK_SCHEMA_VERSION = 1
TOKEN_ENV = "VLM_GPU_LOCK_TOKEN"  # noqa: S105 - env var name, not a secret
PATH_ENV = "VLM_GPU_LOCK_PATH"
_REQUIRED = ("who", "job", "run_id", "host_pid", "started_at", "token")


class GpuLockError(VlmPipelineError):
    """The GPU lock is held by someone else, unreadable, or the caller does not own it."""

    code: ClassVar[str] = "SYS-001"


class GpuLockNotHeldError(GpuLockError):
    """The caller's token does not own the lock (no lock, or another job's lock): a runner/worker protocol error."""

    code: ClassVar[str] = "SYS-004"


@dataclass(frozen=True)
class LockInfo:
    who: str
    job: str
    run_id: str
    host_pid: int
    started_at: str
    token: str
    schema_version: int = LOCK_SCHEMA_VERSION

    def public(self) -> dict[str, object]:
        """Lock content safe to show or log (without the token)."""
        data = asdict(self)
        del data["token"]
        return data


@dataclass(frozen=True)
class LockHandle:
    path: Path
    info: LockInfo


def default_lock_path() -> Path:
    """``$VLM_GPU_LOCK_PATH`` or ``<project_root>/.team/GPU_LOCK`` — the single place that builds this path."""
    override = os.environ.get(PATH_ENV)
    if override:
        return Path(override)
    return project_root() / ".team" / "GPU_LOCK"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    return True


def _parse(raw: str) -> LockInfo:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("lock is not a JSON object")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise KeyError(f"lock is missing fields: {missing}")
    host_pid = data["host_pid"]
    if isinstance(host_pid, bool) or not isinstance(host_pid, int) or host_pid <= 0:
        raise ValueError("host_pid must be a positive integer")
    schema_version = int(data.get("schema_version", LOCK_SCHEMA_VERSION))
    if schema_version > LOCK_SCHEMA_VERSION:
        raise ValueError(f"lock schema_version {schema_version} is newer than supported {LOCK_SCHEMA_VERSION}")
    return LockInfo(
        who=str(data["who"]),
        job=str(data["job"]),
        run_id=str(data["run_id"]),
        host_pid=host_pid,
        started_at=str(data["started_at"]),
        token=str(data["token"]),
        schema_version=schema_version,
    )


# SYS-001 params (error catalog): {who, job, since, state}. No paths/tokens: params reach the UI.
_CORRUPT_PARAMS: dict[str, object] = {"who": "?", "job": "?", "since": "?", "state": "corrupt"}


def _busy_params(info: LockInfo, *, pid_alive: Callable[[int], bool] = _pid_alive) -> dict[str, object]:
    state = "stale" if not pid_alive(info.host_pid) else "busy"
    return {"who": info.who, "job": info.job, "since": info.started_at, "state": state}


def read_lock(path: Path) -> LockInfo | None:
    """Return the current lock, ``None`` if there is none.

    Raises:
        GpuLockError: the lock exists but is unreadable or malformed (callers must treat the GPU as busy).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GpuLockError(f"GPU lock unreadable: {type(exc).__name__}", params=_CORRUPT_PARAMS) from exc
    try:
        return _parse(raw)
    except (ValueError, TypeError, KeyError) as exc:
        raise GpuLockError(f"GPU lock malformed: {type(exc).__name__}", params=_CORRUPT_PARAMS) from exc


def is_alive(info: LockInfo, *, pid_alive: Callable[[int], bool] = _pid_alive) -> bool:
    """Whether the lock holder's host process still exists."""
    return pid_alive(info.host_pid)


def is_stale(info: LockInfo, *, pid_alive: Callable[[int], bool] = _pid_alive) -> bool:
    """A stale lock (dead holder) is only reported; it is never removed by this module."""
    return not is_alive(info, pid_alive=pid_alive)


def acquire(
    path: Path,
    *,
    who: str,
    job: str,
    run_id: str,
    host_pid: int | None = None,
    now: Callable[[], str] = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"),
) -> LockHandle:
    """Create the lock atomically or raise ``GpuLockError`` (with the holder's public info) if it already exists."""
    info = LockInfo(
        who=who,
        job=job,
        run_id=run_id,
        host_pid=host_pid if host_pid is not None else os.getpid(),
        started_at=now(),
        token=secrets.token_hex(16),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(asdict(info), indent=1) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)
    except FileExistsError:
        try:
            current = read_lock(path)
        except GpuLockError as exc:
            raise GpuLockError("GPU busy (lock unreadable)", params=exc.params) from None
        params = _busy_params(current) if current is not None else {**_CORRUPT_PARAMS, "state": "busy"}
        raise GpuLockError("GPU busy", params=params) from None
    finally:
        tmp.unlink(missing_ok=True)
    return LockHandle(path=path, info=info)


def verify_token(path: Path, token: str) -> None:
    """Raise ``GpuLockError`` unless ``path`` holds a lock whose token equals ``token``."""
    current = read_lock(path)
    if current is None:
        raise GpuLockNotHeldError("GPU lock not held")
    if not secrets.compare_digest(current.token.encode(), token.encode()):  # bytes: str form raises on non-ASCII
        raise GpuLockNotHeldError(f"GPU lock held by another job (who={current.who}, run_id={current.run_id})")


def release(handle: LockHandle) -> None:
    """Remove the lock only if it is still ours (token match). Releasing an already-gone lock is a no-op."""
    try:
        verify_token(handle.path, handle.info.token)
    except GpuLockError:
        if read_lock(handle.path) is None:
            return
        raise
    handle.path.unlink()


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """``python -m vlm_pipeline.common.gpu_lock {status|acquire|release}`` for shell scripts and skills.

    ``acquire`` prints the lock JSON (including the token) once; keep the token to release. Never tee that output
    into run logs (``outputs/runs/*/logs``): the token must not outlive the lock on disk. ``--host-pid`` must be
    the PID of the process that stays alive for the whole job (usually the calling shell: ``--host-pid $$``).
    """
    import argparse

    from vlm_pipeline.common.errors import ExitCode

    parser = argparse.ArgumentParser(prog="python -m vlm_pipeline.common.gpu_lock")
    parser.add_argument("--path", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    acq = sub.add_parser("acquire")
    acq.add_argument("--who", required=True)
    acq.add_argument("--job", required=True)
    acq.add_argument("--run-id", required=True)
    acq.add_argument("--host-pid", type=int, required=True)
    rel = sub.add_parser("release")
    rel.add_argument("--token", required=True)
    args = parser.parse_args(argv)
    path: Path = args.path or default_lock_path()

    try:
        if args.cmd == "status":
            info = read_lock(path)
            if info is None:
                print(json.dumps({"held": False}))
                return int(ExitCode.OK)
            print(json.dumps({"held": True, "stale": is_stale(info), **info.public()}))
            return int(ExitCode.LOCK_BUSY)
        if args.cmd == "acquire":
            handle = acquire(path, who=args.who, job=args.job, run_id=args.run_id, host_pid=args.host_pid)
            print(json.dumps(asdict(handle.info)))
            return int(ExitCode.OK)
        info = read_lock(path)
        if info is None:
            return int(ExitCode.OK)
        release(LockHandle(path=path, info=LockInfo(**{**asdict(info), "token": args.token})))
        return int(ExitCode.OK)
    except GpuLockError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc), "params": exc.params}))
        return int(ExitCode.LOCK_BUSY)


if __name__ == "__main__":
    raise SystemExit(main())
