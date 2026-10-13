"""Runs stuck at "running"/"queued" whose process is gone become ``interrupted`` (PM 2026-10-13).

A run.json is written by the process that runs the job; if that process dies (app restart, crash, a script killed),
nobody writes the final status and the UI shows the job running forever. ``reconcile`` (app start-up and every
storage-scheduler pass) marks such a run ``interrupted`` with ``error_code`` SYS-003 and an ``events.jsonl`` line
saying why. A run is left alone while ANY sign of life remains:

* its ``pid`` is alive (a reused PID keeps it "running": the safe side);
* it is the GPU lock's run and the lock holder's PID is alive;
* a container ``vlm-<run_id>`` is running; WITHOUT a docker view (compose app, ADR-015) no run that has a ``pid``
  is judged at all: its pid may belong to another PID namespace or to an app process that restarted while the
  runner still runs the job (bao-cleanup review) — the runner will answer that later;
* it has no ``pid`` yet (still waiting for memory or the lock) and something in its run dir changed within
  ``QUIET_S`` (longer than the longest memory wait, 30 min).

``interrupted`` keeps checkpoints for a resume and retention never deletes it (runinfo.RunStatus).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.runinfo import RunStatus, read_manifest, write_manifest

logger = logging.getLogger(__name__)
RUN_ROOTS = (Path("outputs") / "runs", Path("outputs") / "graph" / "runs")
QUIET_S = 2 * 3600.0
LIVE = (RunStatus.RUNNING, RunStatus.QUEUED)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, another user's
        return True
    return True


def running_containers() -> set[str] | None:
    """Names of running containers, or None when docker cannot be asked (then no run is judged by it)."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def _last_activity(run_dir: Path) -> float:
    """Newest mtime of the run dir, its entries and one level below (logs/train.log, checkpoints/*)."""
    latest = 0.0
    paths = [run_dir]
    for child in run_dir.iterdir():
        paths.append(child)
        if child.is_dir() and not child.is_symlink():
            paths.extend(child.iterdir())
    for path in paths:
        try:
            latest = max(latest, path.lstat().st_mtime)
        except OSError:
            continue
    return latest


class LockUnreadableError(RuntimeError):
    """A malformed lock: whose run it guards is unknown, so no run is judged this pass."""


def lock_run(path: Path | None = None) -> tuple[str | None, int | None]:
    """(run id, holder pid) of the GPU lock at ``path`` (default lock when None)."""
    try:
        info = gpu_lock.read_lock(path or gpu_lock.default_lock_path())
    except gpu_lock.GpuLockError as exc:
        raise LockUnreadableError(str(exc)) from None
    return (info.run_id, info.host_pid) if info is not None else (None, None)


def reconcile(
    root: Path,
    *,
    now: float | None = None,
    alive: Callable[[int], bool] = pid_alive,
    containers: Callable[[], set[str] | None] = running_containers,
    lock_path: Path | None = None,
    lock_reader: Callable[[], tuple[str | None, int | None]] | None = None,
) -> list[str]:
    """Mark dead "running"/"queued" runs interrupted; returns their run ids. Never raises for one bad run."""
    now = time.time() if now is None else now
    try:
        lock_id, lock_pid = lock_reader() if lock_reader is not None else lock_run(lock_path)
    except LockUnreadableError as exc:
        logger.warning("stale-run check skipped: %s", exc)
        return []
    names: set[str] | None = None
    asked = False
    fixed: list[str] = []
    for run_dir, manifest in _live_runs(root):
        if manifest.pid is not None and alive(manifest.pid):
            continue
        if manifest.run_id == lock_id and lock_pid is not None and alive(int(lock_pid)):
            continue
        if manifest.pid is None and now - _last_activity(run_dir) < QUIET_S:
            continue  # may still be waiting for memory or the lock
        if not asked:
            names, asked = containers(), True
            if names is None:
                logger.info("stale-run check: no container view, runs with a pid are not judged")
        if names is None and manifest.pid is not None:
            continue
        if names and (f"vlm-{manifest.run_id}" in names or f"vlm-{run_dir.name}" in names):
            continue
        _mark(run_dir, manifest, now)
        fixed.append(manifest.run_id)
    return fixed


def _live_runs(root: Path) -> Iterator[tuple[Path, Any]]:
    """(run dir, manifest) of every readable run still "running"/"queued" under the run roots."""
    for run_root in RUN_ROOTS:
        base = root / run_root
        for run_dir in sorted(base.iterdir()) if base.is_dir() else []:
            if run_dir.is_symlink() or not (run_dir / "run.json").is_file():
                continue
            try:
                manifest = read_manifest(run_dir)
            except (OSError, ValueError) as exc:
                logger.warning("stale-run check skips %s: %s", run_dir.name, type(exc).__name__)
                continue
            if manifest.status in LIVE:
                yield run_dir, manifest


def _mark(run_dir: Path, manifest: Any, now: float) -> None:
    reason = "process_gone" if manifest.pid is not None else "no_activity"
    was = str(manifest.status)
    write_manifest(run_dir, manifest.model_copy(update={
        "status": RunStatus.INTERRUPTED, "error_code": manifest.error_code or "SYS-003", "ended_at": now}))  # fmt: skip
    event = {"ts": now, "type": "status_reconciled", "from": was, "status": "interrupted", "reason": reason}
    try:  # never through a planted symlink; a refused event must not stop the pass (the status is already fixed)
        fd = os.open(
            run_dir / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644
        )
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except OSError as exc:
        logger.warning("run %s: reconcile event not written (%s)", manifest.run_id, exc.strerror)
    logger.warning("run %s was %s but its process is gone (%s): marked interrupted", manifest.run_id, was, reason)


__all__ = ["QUIET_S", "LockUnreadableError", "lock_run", "reconcile"]
