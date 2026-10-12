"""Clean up gate-run git worktrees under ``<root>/.worktrees/<run_id>`` (ADR-002 A1, ADR-005 D7).

Gate runs execute from ``git worktree add .worktrees/<run_id> <commit>``; their outputs live in
``outputs/runs/<run_id>``, so removing the worktree loses no results. ``plan()`` is read-only; ``apply()`` re-plans
and removes only worktrees still planned for removal, always via ``git worktree remove`` (never ``rm -rf``, never
``--force``: a dirty or locked worktree makes git refuse, which is reported) followed by ``git worktree prune``.

A worktree is removed only when its run is terminal and not resumable: completed, discarded, failed, cancelled or
evicted. queued/running/interrupted, a ``KEEP`` file, the run holding (or an unreadable) GPU lock, a missing run
directory or an unknown status all keep it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vlm_pipeline.artifacts.retention import RUNS_REL, check_project_root, read_run_status
from vlm_pipeline.common.diskguard import LockStatus, check_lock
from vlm_pipeline.common.runinfo import RunStatus

WORKTREES_REL = Path(".worktrees")
REMOVABLE_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.DISCARDED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.EVICTED}
)

# never execute repo-configured programs (fsmonitor, hooks) while git checks whether a worktree is dirty
_SAFE_GIT_CONFIG = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null")

# (argv after "git -C <root>") -> (returncode, stdout+stderr)
GitRunner = Callable[[Sequence[str]], tuple[int, str]]


def _is_allowed_git(argv: Sequence[str]) -> bool:
    """Only ``worktree list --porcelain``, ``worktree remove <path>`` (no --force) and ``worktree prune``."""
    if len(argv) < 2 or argv[0] != "worktree":
        return False
    sub, rest = argv[1], list(argv[2:])
    if sub == "list":
        return rest == ["--porcelain"]
    if sub == "remove":
        return len(rest) == 1 and not rest[0].startswith("-")
    return sub == "prune" and rest == []


def _clean_env() -> dict[str, str]:
    """Inherited env without GIT_* (GIT_DIR, GIT_WORK_TREE, ... would redirect ``git -C root`` to another repo)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git_runner(root: Path) -> GitRunner:
    def run(argv: Sequence[str]) -> tuple[int, str]:
        if not _is_allowed_git(argv):
            raise ValueError(f"worktree cleanup only runs list/remove/prune, got: {list(argv)}")
        git = shutil.which("git")
        if git is None:
            return 127, "git not found"
        out = subprocess.run(  # noqa: S603 - fixed, validated argv; no shell
            [git, *_SAFE_GIT_CONFIG, "-C", str(root), *argv],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=_clean_env(),
        )
        return out.returncode, (out.stdout + out.stderr).strip()

    return run


@dataclass(frozen=True)
class WorktreeItem:
    run_id: str
    path: str  # absolute worktree path as git reports it
    status: str
    action: str  # "remove" | "keep"
    reason: str
    blocked_by: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "blocked_by": self.blocked_by,
        }


@dataclass
class WorktreeResult:
    removed: list[WorktreeItem] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (run_id, git message)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _listed_worktrees(run: GitRunner) -> list[Path]:
    code, out = run(["worktree", "list", "--porcelain"])
    if code != 0:
        raise RuntimeError(f"git worktree list failed: {out}")
    return [Path(line.removeprefix("worktree ")) for line in out.splitlines() if line.startswith("worktree ")]


def _decide(root: Path, run_id: str, lock: LockStatus | None) -> tuple[str, str, str | None]:
    """(status, reason, blocked_by) for the worktree of ``run_id``."""
    run_dir = root / RUNS_REL / run_id
    if not run_dir.is_dir():
        return "unknown", "no outputs/runs/<run_id> for this worktree", "UNKNOWN_RUN"
    status, _ended = read_run_status(run_dir)
    if (run_dir / "KEEP").exists():
        return status, "run is pinned", "PINNED_OR_KEEP"
    if lock is not None and (lock.corrupt or lock.run_id is None or lock.run_id == run_id):
        return status, "run holds the GPU lock (or the lock is unreadable)", "GPU_LOCK_HOLDER"
    if status not in {s.value for s in RunStatus}:
        return status, "missing/unknown run status", "UNKNOWN_STATUS"
    if status not in REMOVABLE_STATUSES:
        return status, f"run is {status}", "RUN_NOT_FINISHED"
    return status, f"run is {status}; outputs live in outputs/runs", None


def plan(
    root: Path,
    *,
    run: GitRunner | None = None,
    run_ids: Collection[str] | None = None,
    lock_reader: Callable[[Path], LockStatus | None] = check_lock,
    lock_path: Path | None = None,
) -> list[WorktreeItem]:
    """Read-only: one item per git worktree located directly under ``<root>/.worktrees``."""
    git = run if run is not None else git_runner(root)
    lock = lock_reader(lock_path if lock_path is not None else root / ".team" / "GPU_LOCK")
    base = (root / WORKTREES_REL).resolve()
    items: list[WorktreeItem] = []
    for path in _listed_worktrees(git):
        if path.parent != base or (run_ids is not None and path.name not in run_ids):
            continue  # the main checkout and anything outside .worktrees/ are never candidates
        status, reason, blocked = _decide(root, path.name, lock)
        items.append(WorktreeItem(path.name, str(path), status, "keep" if blocked else "remove", reason, blocked))
    return items


def apply(
    root: Path,
    run_ids: Collection[str],
    *,
    by: str,
    run: GitRunner | None = None,
    lock_reader: Callable[[Path], LockStatus | None] = check_lock,
    lock_path: Path | None = None,
) -> WorktreeResult:
    """Remove the worktrees of ``run_ids`` that are still planned for removal, then ``git worktree prune``."""
    check_project_root(root)
    git = run if run is not None else git_runner(root)
    fresh = {i.run_id: i for i in plan(root, run=git, lock_reader=lock_reader, lock_path=lock_path)}
    result = WorktreeResult()
    for run_id in sorted(set(run_ids)):
        item = fresh.get(run_id)
        if item is None or item.action != "remove":
            why = (item.blocked_by or "not planned") if item else "no such worktree"
            result.skipped.append((run_id, why))
            continue
        code, out = git(["worktree", "remove", item.path])
        if code != 0:  # dirty/locked worktree: git refuses; we never --force
            result.failed.append((run_id, out))
            continue
        result.removed.append(item)
        _log(root, {"event": "worktree_removed", "run_id": run_id, "path": item.path, "by": by})
    git(["worktree", "prune"])
    return result


def _log(root: Path, record: dict[str, object]) -> None:
    path = root / "outputs" / "system" / "cleanup.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {**record, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ts_epoch": time.time(), "bytes": 0}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
