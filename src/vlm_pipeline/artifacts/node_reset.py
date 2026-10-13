"""Reset one graph step (G-4, PM 2026-10-13): delete every attempt dir of a node to run it again. Owner: bao-cleanup.

A node ``<nid>-<key16>`` (``run_claims.NODE_BASE``) gets one run dir per try (P017): ``<base>[-g<N>][-a<M>]``. After
10 tries the executor stops (SYS-005). Claims are never removed, so deleting the dirs alone would not let it run
again: ``apply_reset`` also bumps the node's generation (``outputs/system/node_generations/<base>.json``, read by the
executor through ``run_claims.node_generation``) and invalidates the node's cache entries.

``plan_reset`` is read-only. It refuses (``blocked``) while any attempt may still be in use:

* an attempt is running/queued, holds the GPU lock, or the lock is unreadable;
* an attempt has a resume lease (the executor is about to start one) or another node's attempt resumes from it;
* a cache entry of the node is leased by a running graph, or an attempt has a ``KEEP`` file.

An attempt with checkpoints makes the reset a PERMANENT_LOSS: the caller requires the node name typed. Run claims,
leases and every file outside ``outputs/graph/runs`` are never touched.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from vlm_pipeline.artifacts import retention
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.artifacts.storage import du_bytes
from vlm_pipeline.common.diskguard import LockStatus, check_lock
from vlm_pipeline.common.run_claims import GENERATIONS_REL, MAX_GENERATION, NODE_BASE, GenerationError, node_generation
from vlm_pipeline.common.runinfo import RunStatus

PERMANENT_LOSS = "PERMANENT_LOSS"
LIVE = frozenset({RunStatus.RUNNING, RunStatus.QUEUED})


@dataclass(frozen=True)
class Attempt:
    name: str
    status: str
    reclaim_bytes: int
    has_checkpoint: bool


@dataclass(frozen=True)
class ResetPlan:
    node: str
    attempts: tuple[Attempt, ...]
    cache_keys: tuple[str, ...]
    generation: int  # current; a reset writes generation + 1
    blocked_by: str | None = None
    blocked_params: dict[str, str] = field(default_factory=dict)

    @property
    def reclaim_bytes(self) -> int:
        return sum(a.reclaim_bytes for a in self.attempts)

    @property
    def consequence_code(self) -> str | None:
        return PERMANENT_LOSS if any(a.has_checkpoint for a in self.attempts) else None


def _attempt_re(node: str) -> re.Pattern[str]:
    # every generation, attempt and node job dir (``-model``/``-base``) of exactly this node
    return re.compile(rf"{re.escape(node)}(?:-g[0-9]{{1,3}})?(?:-a[0-9]{{1,2}})?(?:-[a-z0-9_]{{1,32}})?", re.ASCII)


def attempt_dirs(root: Path, node: str) -> list[Path]:
    """Real attempt dirs of ``node`` directly in ``outputs/graph/runs`` (symlinks are never listed or followed)."""
    if not NODE_BASE.fullmatch(node):
        raise ValueError("bad_node")
    base = root / retention.GRAPH_RUNS_REL
    if base.is_symlink() or not base.is_dir():
        return []
    pattern = _attempt_re(node)
    return [p for p in sorted(base.iterdir()) if pattern.fullmatch(p.name) and p.is_dir() and not p.is_symlink()]


def _has_checkpoint(run_dir: Path) -> bool:
    ckpts = run_dir / "checkpoints"
    if ckpts.is_symlink() or not ckpts.is_dir():
        return False
    return any(p.name.startswith(retention.CKPT_PREFIX) and p.is_dir() for p in ckpts.iterdir())


def _blocked(  # noqa: PLR0911 - flat table of refusals, first match wins
    root: Path,
    node: str,
    dirs: list[Path],
    *,
    statuses: dict[str, str],
    keys: list[str],
    index: ArtifactIndex,
    lock: LockStatus | None,
) -> tuple[str | None, dict[str, str]]:
    names = {d.name for d in dirs}
    live = sorted(n for n, s in statuses.items() if s in LIVE)
    if live:
        return "RUN_NOT_FINISHED", {"run": live[0]}
    unknown = sorted(n for n, s in statuses.items() if s not in {r.value for r in RunStatus})
    if unknown:  # unreadable/missing run.json: maybe still running (retention keeps these too)
        return "UNKNOWN_STATUS", {"run": unknown[0]}
    if lock is not None and (lock.corrupt or lock.run_id is None or lock.run_id in names):
        return "GPU_LOCK_HOLDER", {"run": lock.run_id or "?"}
    leases = retention.resume_lease_files(root)
    if leases is None or any(p.name.startswith(node) for p in leases):  # an attempt is starting, or unknown
        return "RUN_NOT_FINISHED", {"run": node}
    inputs = retention.resume_inputs(root, ignore_node=node)  # its own attempts resuming from each other do not count
    held = [d.name for d in dirs if inputs.blocks(d)]
    if held:
        return retention.INPUT_REASON, {"run": held[0]}
    for key in keys:
        users = index.leased_by(key)
        if users:
            return "IN_USE_BY_RUNNING_WORKFLOW", {"workflow": users[0]}
    kept = [d.name for d in dirs if (d / "KEEP").exists()]
    if kept:
        return "PINNED_OR_KEEP", {"run": kept[0]}
    return None, {}


def plan_reset(root: Path, index: ArtifactIndex, node: str, *, lock_path: Path | None = None) -> ResetPlan:
    """Read-only: what resetting ``node`` would delete, and why it is refused (``blocked_by``)."""
    dirs = attempt_dirs(root, node)
    statuses = {d.name: retention.read_run_status(d)[0] for d in dirs}
    attempts = tuple(Attempt(d.name, statuses[d.name], du_bytes(d), _has_checkpoint(d)) for d in dirs)
    keys = index.keys_for_run_dirs(dirs)
    lock = check_lock(lock_path if lock_path is not None else root / ".team" / "GPU_LOCK")
    try:
        generation = node_generation(root, node)
    except GenerationError:
        return ResetPlan(node, attempts, tuple(keys), 0, "GENERATION_UNREADABLE", {"run": node})
    if generation >= MAX_GENERATION:
        return ResetPlan(node, attempts, tuple(keys), generation, "GENERATION_LIMIT", {"run": node})
    blocked, params = _blocked(root, node, dirs, statuses=statuses, keys=keys, index=index, lock=lock)
    return ResetPlan(node, attempts, tuple(keys), generation, blocked, params)


def _write_generation(root: Path, node: str, generation: int, by: str) -> None:
    """``outputs/system/node_generations/<node>.json``; no symlink anywhere on the path, written in place."""
    folder = root / GENERATIONS_REL
    for part in (root / "outputs", root / "outputs" / "system"):
        if part.is_symlink() or not part.is_dir():
            raise OSError(f"{part} is not a plain directory")
    folder.mkdir(mode=0o755, exist_ok=True)
    st = folder.lstat()
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"{folder} is not a plain directory")
    body = json.dumps({"gen": generation, "ts": time.time(), "by": by}).encode()
    fd = os.open(folder / f"{node}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(body)


@dataclass
class ResetResult:
    deleted: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    generation: int = 0
    error: str | None = None  # a later step failed after deletes (still audited)


def apply_reset(
    root: Path,
    index: ArtifactIndex,
    node: str,
    *,
    by: str,
    expected: ResetPlan,
    approved_by: str | None = None,
    lock_path: Path | None = None,
) -> ResetResult:
    """Re-plan, then delete every attempt dir, invalidate the node's cache entries and bump its generation.

    The caller holds ``retention.cleanup_lock`` and has checked the typed confirmation against ``expected``.

    Raises:
        PermissionError: the fresh plan is blocked or differs from ``expected`` (the user must look again).
    """
    retention.check_project_root(root)
    fresh = plan_reset(root, index, node, lock_path=lock_path)
    if fresh.blocked_by is not None:
        raise PermissionError(fresh.blocked_by)
    if {a.name for a in fresh.attempts} != {a.name for a in expected.attempts} or (
        fresh.consequence_code != expected.consequence_code
    ):
        raise PermissionError("STATE_CHANGED")
    # The executor does not take cleanup_lock: it could lease a source between this re-plan and the deletes (ms). The
    # worst case is that new attempt failing at start; no other dir is ever deleted (security N2, accepted).
    base = (root / retention.GRAPH_RUNS_REL).resolve()
    result = ResetResult(generation=fresh.generation + 1)
    try:
        _delete_attempts(base, fresh, result)
        for key in fresh.cache_keys:
            index.invalidate(key)
        _write_generation(root, node, result.generation, by)  # also after a partial delete: new names next run
    except OSError as exc:
        result.error = f"{type(exc).__name__}: {exc.strerror or exc}"
        raise
    finally:  # every real delete leaves an audit line, also when a later step failed (security R2)
        _audit(root, node, fresh, result, by=by, approved_by=approved_by)
    return result


def _delete_attempts(base: Path, plan: ResetPlan, result: ResetResult) -> None:
    for attempt in plan.attempts:
        target = base / attempt.name
        if target.is_symlink() or target.resolve().parent != base:
            result.failed.append(attempt.name)
            continue
        try:
            shutil.rmtree(target)
        except OSError:
            result.failed.append(attempt.name)
            continue
        result.deleted.append(attempt.name)
        result.freed_bytes += attempt.reclaim_bytes


def _audit(root: Path, node: str, plan: ResetPlan, result: ResetResult, *, by: str, approved_by: str | None) -> None:
    now = time.time()
    retention.append_jsonl(
        root / retention.CLEANUP_LOG_REL,
        {
            "event": "node_reset",
            "node": node,
            "deleted": result.deleted,
            "failed": result.failed,
            "bytes": result.freed_bytes,
            "generation": result.generation,
            "error": result.error,
            "permanent_loss": plan.consequence_code == PERMANENT_LOSS,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            "ts_epoch": now,
            "by": by,
            "manual": True,
            "approved_by": approved_by,
        },
    )


def unapproved_reset_bytes_since(log: Path, since: float) -> int:
    """Bytes of node resets without ``approved_by`` after ``since`` (shared-host approval threshold)."""
    if not log.exists():
        return 0
    total = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("event") == "node_reset" and not rec.get("approved_by") and float(rec["ts_epoch"]) >= since:
                total += int(rec.get("bytes", 0))
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return total


__all__ = ["PERMANENT_LOSS", "Attempt", "ResetPlan", "ResetResult", "apply_reset", "attempt_dirs", "plan_reset"]
