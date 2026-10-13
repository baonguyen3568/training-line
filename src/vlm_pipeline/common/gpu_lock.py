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
import logging
import math
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
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
    # The job's run dir relative to the project/data root ("outputs/graph/runs/<id>"), so tools (E2E preflight,
    # watchdog, dashboard) need not search for it (tester 2026-10-13). Optional: older locks and manual locks have none.
    run_dir: str | None = None

    def public(self) -> dict[str, object]:
        """Lock content safe to show or log (without the token)."""
        data = asdict(self)
        del data["token"]
        return data


@dataclass(frozen=True)
class LockHandle:
    path: Path
    info: LockInfo


LOCK_REL = Path(".team") / "lock" / "GPU_LOCK"  # its own folder: compose mounts exactly it (security, PM 2026-10-13)
LEGACY_REL = Path(".team") / "GPU_LOCK"  # before 2026-10-13: a lock still there means BUSY (never two locks at once)
# TODO(bao-backend, transition end): once app 7870 and every job run post-2026-10-13 code, drop LEGACY_REL, the
# acquire guard, the old-path mirror (_link_legacy / _release_legacy) and lock_present's legacy branch (BOARD G).


def lock_path_for(root: Path) -> Path:
    """The GPU lock of a project/data root. Every module builds the path here (no ``.team/...`` literals elsewhere)."""
    return root / LOCK_REL


def default_lock_path() -> Path:
    """``$VLM_GPU_LOCK_PATH`` or ``<project_root>/.team/lock/GPU_LOCK``."""
    override = os.environ.get(PATH_ENV)
    if override:
        return Path(override)
    return lock_path_for(project_root())


def _legacy_path(path: Path) -> Path | None:
    """The pre-move lock path (``.team/GPU_LOCK``) of a ``.team/lock/GPU_LOCK`` path, existing or not.

    Only a parent named ``lock`` qualifies: compose's ``/run/vlm-lock/GPU_LOCK`` never had a legacy file (by design).
    """
    if path.name == "GPU_LOCK" and path.parent.name == "lock":
        return path.parent.parent / "GPU_LOCK"
    return None


def _legacy_of(path: Path) -> Path | None:
    """The pre-move lock file next to ``path``, when it still exists (transition guard)."""
    legacy = _legacy_path(path)
    return legacy if legacy is not None and os.path.lexists(legacy) else None


def lock_present(path: Path) -> bool:
    """True when the lock file (or, during the move, the legacy ``.team/GPU_LOCK``) exists, even as a symlink.

    For callers that skip work while the GPU is busy without parsing the lock: same truth as ``acquire``'s guard.
    """
    return os.path.lexists(path) or _legacy_of(path) is not None


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
        run_dir=_run_dir(data.get("run_dir")),
    )


def _run_dir(value: object) -> str | None:
    """A relative path under outputs/ without '..' or None (anything else in the file is ignored, never trusted)."""
    if not isinstance(value, str) or len(value) > 512:
        return None
    parts = PurePosixPath(value).parts
    if value.startswith("/") or ".." in parts or not parts or parts[0] != "outputs" or "\\" in value:
        return None
    return value


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
    legacy = _legacy_of(path) if not os.path.lexists(path) else None
    if legacy is not None:  # a job started before the move still holds the old file: report it, it is the GPU's holder
        path = legacy
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


RESERVED_NAME = "GPU_RESERVED"  # next to the lock: the next turn belongs to one name (PM, maintenance)
RESERVATIONS_LOG_NAME = "GPU_RESERVATIONS.jsonl"  # who reserved, why, until when (append only)
MAX_RESERVE_S = 30 * 60  # PM: enough for maintenance; longer would be a disguised way to hold the GPU


@dataclass(frozen=True)
class Reservation:
    who: str
    reason: str
    created_at: float
    until: float


def reservation_path(lock_path: Path) -> Path:
    return lock_path.with_name(RESERVED_NAME)


def read_reservation(lock_path: Path, *, now: float | None = None) -> Reservation | None:
    """The reservation in force (an expired or unreadable one counts as none: it can never wedge the GPU)."""
    from vlm_pipeline.common.runinfo import read_small_json  # never follows a link, never blocks on a FIFO

    try:
        data = read_small_json(reservation_path(lock_path), max_bytes=4096)
        if not isinstance(data, dict):
            return None
        res = Reservation(who=str(data["who"]), reason=str(data["reason"]), created_at=float(data["created_at"]),
                          until=float(data["until"]))  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError):
        return None
    t = time.time() if now is None else now
    if not (math.isfinite(res.created_at) and math.isfinite(res.until)) or (
        res.until - res.created_at > MAX_RESERVE_S + 5 or res.created_at > t + 60
    ):  # not written by reserve(): a hand-made or job-made file never holds the GPU for one name
        logging.getLogger(__name__).warning("GPU reservation file ignored: outside the %ss limit", MAX_RESERVE_S)
        return None
    return res if res.until > t else None


def reserve(lock_path: Path, *, who: str, reason: str, minutes: float, now: float | None = None) -> Reservation:
    """Keep the next GPU turn for ``who`` (at most 30 min); refused while another name holds a reservation."""
    if not who.strip() or not reason.strip():
        raise ValueError("a reservation needs who and a reason")
    if not 0 < minutes * 60 <= MAX_RESERVE_S:
        raise ValueError(f"a reservation lasts at most {MAX_RESERVE_S // 60} minutes")
    t = time.time() if now is None else now
    current = read_reservation(lock_path, now=t)
    if current is not None and current.who != who:
        raise GpuLockError("GPU already reserved", params={"who": current.who, "until": current.until})
    res = Reservation(who=who, reason=reason, created_at=t, until=t + minutes * 60)
    path = reservation_path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(res)) + "\n")
    try:
        if current is None:  # none in force: create it atomically, so of two names reserving at once one is refused
            # An expired or unreadable one goes first; two names racing within this instant can still both pass
            # (maintenance is rare: accepted, cleanup N1).
            path.unlink(missing_ok=True)
            try:
                os.link(tmp, path)
            except FileExistsError:
                won = read_reservation(lock_path, now=t)
                raise GpuLockError("GPU already reserved", params={"who": won.who if won else "?"}) from None
        else:
            tmp.replace(path)  # extending our own reservation
    finally:
        tmp.unlink(missing_ok=True)
    with (path.parent / RESERVATIONS_LOG_NAME).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "reserve", **asdict(res)}) + "\n")
    return res


def unreserve(lock_path: Path, *, who: str) -> bool:
    """Drop ``who``'s reservation (another name's is never touched); True when one was removed."""
    current = read_reservation(lock_path)
    if current is None or current.who != who:
        return False
    reservation_path(lock_path).unlink(missing_ok=True)
    with (lock_path.parent / RESERVATIONS_LOG_NAME).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "unreserve", "who": who, "at": time.time()}) + "\n")
    return True


def acquire(
    path: Path,
    *,
    who: str,
    job: str,
    run_id: str,
    host_pid: int | None = None,
    run_dir: str | None = None,
    now: Callable[[], str] = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"),
) -> LockHandle:
    """Create the lock atomically or raise ``GpuLockError`` (with the holder's public info) if it already exists.

    Transition (both directions, PM 2026-10-13): a lock at the old path (``.team/GPU_LOCK``) means busy, and a new
    lock is ALSO linked there (same inode, ``O_EXCL``) so processes still running the old code see the GPU busy too.
    The old path is the shared arbiter: if an old-code job takes it between our two links, ours is rolled back.
    """
    reserved = read_reservation(path)
    if reserved is not None and reserved.who != who:  # maintenance: the next turn is someone else's
        raise GpuLockError("GPU reserved", params={"who": reserved.who, "job": reserved.reason,
                                                   "since": reserved.created_at, "state": "reserved"})  # fmt: skip
    legacy = _legacy_of(path)
    if legacy is not None:
        try:
            held = read_lock(legacy)
        except GpuLockError as exc:
            raise GpuLockError("GPU busy (old lock unreadable)", params=exc.params) from None
        raise GpuLockError("GPU busy (old lock path)", params=_busy_params(held) if held else dict(_CORRUPT_PARAMS))
    info = LockInfo(
        who=who,
        job=job,
        run_id=run_id,
        host_pid=host_pid if host_pid is not None else os.getpid(),
        started_at=now(),
        token=secrets.token_hex(16),
        run_dir=_run_dir(run_dir),
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
        params = _busy_params(current) if current is not None else dict(_CORRUPT_PARAMS)  # vanished mid-race
        raise GpuLockError("GPU busy", params=params) from None
    try:
        _link_legacy(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    if reserved is not None:
        unreserve(path, who=who)  # the reserved turn is taken: everyone queues normally again after it
    return LockHandle(path=path, info=info)


def _link_legacy(tmp: Path, path: Path) -> None:
    """Mirror a just-created lock at the old path; on a race with old code, undo ours and report that holder."""
    legacy = _legacy_path(path)
    if legacy is None:
        return
    try:
        os.link(tmp, legacy)
    except FileExistsError:
        path.unlink(missing_ok=True)  # ours, created a moment ago: the old-path holder won
        try:
            held = read_lock(legacy)
        except GpuLockError as exc:
            raise GpuLockError("GPU busy (old lock unreadable)", params=exc.params) from None
        busy = _busy_params(held) if held else dict(_CORRUPT_PARAMS)
        raise GpuLockError("GPU busy (old lock path)", params=busy) from None
    except OSError:
        path.unlink(missing_ok=True)  # e.g. no write access to .team/: never hold a lock old code cannot see
        raise


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
            _release_legacy(handle)
            return
        raise
    handle.path.unlink()
    _release_legacy(handle)


def _release_legacy(handle: LockHandle) -> None:
    """Remove the old-path mirror of our lock (token match only: an old-code holder's lock is never touched)."""
    legacy = _legacy_path(handle.path)
    if legacy is None or not os.path.lexists(legacy):
        return
    try:
        held = _parse(legacy.read_text(encoding="utf-8")) if not legacy.is_symlink() else None
    except (OSError, ValueError, TypeError, KeyError):  # not a lock this module wrote: leave it alone
        return
    if held is not None and secrets.compare_digest(held.token.encode(), handle.info.token.encode()):
        legacy.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one return per subcommand
    """``python -m vlm_pipeline.common.gpu_lock {status|acquire|release|reserve|unreserve}`` for scripts and skills.

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
    res = sub.add_parser("reserve", help="keep the next GPU turn for --who (maintenance, max 30 min)")
    res.add_argument("--who", required=True)
    res.add_argument("--reason", required=True)
    res.add_argument("--minutes", type=float, required=True)
    unres = sub.add_parser("unreserve")
    unres.add_argument("--who", required=True)
    args = parser.parse_args(argv)
    path: Path = args.path or default_lock_path()

    try:
        if args.cmd == "status":
            info = read_lock(path)
            reserved = read_reservation(path)
            if info is None:
                print(json.dumps({"held": False, **({"reserved": asdict(reserved)} if reserved else {})}))
                return int(ExitCode.OK)
            print(json.dumps({"held": True, "stale": is_stale(info), **info.public()}))
            return int(ExitCode.LOCK_BUSY)
        if args.cmd == "reserve":
            print(json.dumps(asdict(reserve(path, who=args.who, reason=args.reason, minutes=args.minutes))))
            return int(ExitCode.OK)
        if args.cmd == "unreserve":
            print(json.dumps({"removed": unreserve(path, who=args.who)}))
            return int(ExitCode.OK)
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
    except ValueError as exc:  # reserve: missing reason, too long
        print(json.dumps({"error": "bad_request", "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
