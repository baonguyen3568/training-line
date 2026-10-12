"""Checkpoint retention for ``outputs/runs/<run_id>/checkpoints`` (ADR-005 D3 + D7).

``plan()`` is read-only and returns what *would* happen. ``apply()`` re-plans and only executes items that are still
planned the same way (no time-of-check/time-of-use gap), then logs ``checkpoint_pruned`` events. Nothing outside
``<root>/outputs/runs`` is ever touched, symlinks are never followed or removed, and whitelisted files
(run.json, *.jsonl, logs/, export/, ...) are never candidates: only ``epoch_*_step_*`` directories and their
``optim/``, ``dataloader/``, ``rng/`` sub-directories are.

Rules (first match wins, per checkpoint; statuses are ``common.runinfo.RunStatus``):
1. run has a ``KEEP`` file, or is the run holding the GPU lock, or the lock is unreadable -> keep (blocked)
2. status is queued/running/interrupted/evicted -> keep (blocked); missing/unknown status -> keep (UNKNOWN_STATUS)
3. failed/cancelled: nothing before ``failed_grace_days`` after the run's last activity; then ``.incomplete`` and
   non-LATEST checkpoints are deleted, and LATEST is deleted ``failed_latest_days`` after the last activity.
   Last activity = max(run.json ``ended_at``, run.json mtime, newest mtime under ``checkpoints/``), so a wrong or
   back-dated ``ended_at`` can only delay cleanup, never bring it forward.
4. completed/discarded: checkpoint has ``.incomplete`` (AutoModel: save never finished) -> delete
5. target of ``LATEST`` -> keep for ``latest_full_days``, then strip optimizer state
6. target of ``LOWEST_VAL`` -> strip optimizer state
7. any other checkpoint -> delete once older than ``other_keep_days``

Age is the checkpoint directory mtime. Stripping sub-directories updates that mtime, so a stripped checkpoint
starts a new window; this errs on the side of keeping data longer.

A run.json status that is not a ``RunStatus`` value is reported as ``UNKNOWN_STATUS`` (not silently kept as
"not finished"), so a vocabulary mismatch with the run writer shows up in every dry-run.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import tomllib
from collections.abc import Callable, Collection, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from vlm_pipeline.artifacts.storage import du_bytes
from vlm_pipeline.common.diskguard import LockStatus, check_lock
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import RunStatus

Action = Literal["keep", "strip_optim", "delete"]

PLAN_SCHEMA_VERSION = 1
RUNS_REL = Path("outputs") / "runs"
CKPT_PREFIX = "epoch_"
ROLE_LINKS = ("LATEST", "LOWEST_VAL")
STRIP_SUBDIRS = ("optim", "dataloader", "rng")  # resume-only state; model weights stay
CLEANABLE_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.DISCARDED})
FAILED_STATUSES = frozenset({RunStatus.FAILED, RunStatus.CANCELLED})  # cleaned after a grace period (ADR-005)
DAY_S = 86_400.0
PROJECT_NAME = "vlm-pipeline"  # [project].name in pyproject.toml; guards against a wrong VLM_PROJECT_ROOT
CLEANUP_LOG_REL = Path("outputs") / "system" / "cleanup.jsonl"


@dataclass(frozen=True)
class RetentionPolicy:
    """ADR-005 D3 (team CLI defaults)."""

    latest_full_days: float = 7.0
    other_keep_days: float = 7.0
    failed_grace_days: float = 3.0  # failed/cancelled: untouched for this long after the run ended
    failed_latest_days: float = 7.0  # failed/cancelled: LATEST deleted (not stripped) after this long
    approval_threshold_gb: float = 20.0  # D7.4: above this the CLI needs --approved-by
    approval_window_s: float = DAY_S  # unapproved deletes are summed over this window (no splitting into batches)


DEFAULT_POLICY = RetentionPolicy()


@dataclass(frozen=True)
class PlanItem:
    run_id: str
    path: str  # relative to the run dir, e.g. "checkpoints/epoch_0_step_9"
    kind: str
    roles: tuple[str, ...]
    action: Action
    reclaim_bytes: int
    rule: str
    reason: str
    blocked_by: str | None = None

    @property
    def item_id(self) -> str:
        return f"{self.run_id}/{self.path}"

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["roles"] = list(self.roles)
        data["item_id"] = self.item_id
        data["reclaim_gb"] = round(self.reclaim_bytes / GIB, 3)
        return data


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    status: str
    keep: bool
    items: tuple[PlanItem, ...]


@dataclass(frozen=True)
class Plan:
    ts: float
    runs: tuple[RunPlan, ...]
    policy: RetentionPolicy
    lock: LockStatus | None

    @property
    def actionable(self) -> list[PlanItem]:
        return [i for r in self.runs for i in r.items if i.action != "keep" and i.blocked_by is None]

    @property
    def total_reclaim_bytes(self) -> int:
        return sum(i.reclaim_bytes for i in self.actionable)

    @property
    def needs_approval(self) -> bool:
        return self.total_reclaim_bytes / GIB > self.policy.approval_threshold_gb

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.ts)),
            "mode": "dry_run",
            "total_reclaim_gb": round(self.total_reclaim_bytes / GIB, 3),
            "needs_approval": self.needs_approval,
            "approval_reason": "TOTAL_OVER_THRESHOLD" if self.needs_approval else None,
            "threshold_gb": self.policy.approval_threshold_gb,
            "runs": [
                {"run_id": r.run_id, "status": r.status, "keep": r.keep, "items": [i.as_dict() for i in r.items]}
                for r in self.runs
            ],
        }


@dataclass
class ApplyResult:
    done: list[PlanItem] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (item_id, why)

    @property
    def freed_bytes(self) -> int:
        return sum(i.reclaim_bytes for i in self.done)


def _run_status(run_dir: Path) -> tuple[str, float | None]:
    """(status, ended_at) from run.json; status is "unknown"/"unreadable" when it cannot be read."""
    path = run_dir / "run.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "unknown", None
    except (OSError, ValueError):
        return "unreadable", None
    if not isinstance(data, dict):
        return "unreadable", None
    status = data.get("status")
    ended = data.get("ended_at")
    ended_at = float(ended) if isinstance(ended, int | float) and not isinstance(ended, bool) else None
    if ended_at is None:
        ended_at = path.stat().st_mtime  # runner rewrites run.json when the status changes
    return (str(status) if status else "unknown"), ended_at


def _newest_mtime(path: Path) -> float:
    """Newest lstat mtime of ``path`` and everything below it (symlinks are not followed)."""
    try:
        newest = path.lstat().st_mtime
    except OSError:
        return 0.0
    if path.is_symlink() or not path.is_dir():
        return newest
    for dirpath, dirnames, filenames in os.walk(path, onerror=lambda _err: None):
        for name in (*dirnames, *filenames):
            try:
                newest = max(newest, (Path(dirpath) / name).lstat().st_mtime)
            except OSError:
                continue
    return newest


def _roles(ckpt_root: Path) -> dict[str, list[str]]:
    """Map checkpoint dir name -> role names, from top-level symlinks resolving inside ``ckpt_root``."""
    roles: dict[str, list[str]] = {}
    for name in ROLE_LINKS:
        link = ckpt_root / name
        if not link.is_symlink():
            continue
        target = link.resolve()
        if target.parent == ckpt_root.resolve():
            roles.setdefault(target.name, []).append(name)
    return roles


def _strip_bytes(ckpt: Path) -> int:
    return sum(du_bytes(ckpt / sub) for sub in STRIP_SUBDIRS if (ckpt / sub).is_dir() and not (ckpt / sub).is_symlink())


def _plan_checkpoint(  # noqa: PLR0911 - one return per ADR-005 D3 rule, kept as a flat decision table
    run_id: str,
    ckpt: Path,
    roles: list[str],
    *,
    blocked: str | None,
    failed_ended_at: float | None,
    now: float,
    policy: RetentionPolicy,
) -> PlanItem:
    """``failed_ended_at`` is the run's last activity, set only for failed/cancelled runs."""
    rel = f"checkpoints/{ckpt.name}"
    age_days = (now - ckpt.lstat().st_mtime) / DAY_S

    def item(action: Action, nbytes: int, rule: str, reason: str, blocked_by: str | None = None) -> PlanItem:
        return PlanItem(run_id, rel, "checkpoint", tuple(roles), action, nbytes, rule, reason, blocked_by)

    if blocked is not None:
        return item("keep", 0, "D7.blocked", f"protected: {blocked}", blocked)
    if failed_ended_at is not None:
        ended_days = (now - failed_ended_at) / DAY_S
        if ended_days < policy.failed_grace_days:
            return item("keep", 0, "D3.failed_grace", f"failed/cancelled < {policy.failed_grace_days:g} days ago")
        if "LATEST" in roles and not (ckpt / ".incomplete").exists():
            if ended_days < policy.failed_latest_days:
                return item("keep", 0, "D3.failed_latest_wait", f"LATEST kept {policy.failed_latest_days:g} days")
            return item("delete", du_bytes(ckpt), "D3.failed_latest_delete", "failed/cancelled run: LATEST expired")
        return item("delete", du_bytes(ckpt), "D3.failed_other", "failed/cancelled run: not LATEST or incomplete")
    if (ckpt / ".incomplete").exists():
        return item("delete", du_bytes(ckpt), "D3.incomplete", "save never finished (.incomplete)")
    if "LATEST" in roles:
        if age_days < policy.latest_full_days:
            return item("keep", 0, "D3.latest_full", f"LATEST kept whole for {policy.latest_full_days:g} days")
        nbytes = _strip_bytes(ckpt)
        if nbytes == 0:
            return item("keep", 0, "D3.latest_stripped", "LATEST, no optimizer state left")
        return item("strip_optim", nbytes, "D3.latest_after_days", "LATEST older than window: strip optimizer only")
    if "LOWEST_VAL" in roles:
        nbytes = _strip_bytes(ckpt)
        if nbytes == 0:
            return item("keep", 0, "D3.best_stripped", "best checkpoint, no optimizer state left")
        return item("strip_optim", nbytes, "D3.best_strip", "best checkpoint: keep weights, strip optimizer")
    if age_days < policy.other_keep_days:
        return item("keep", 0, "D3.other_wait", f"younger than {policy.other_keep_days:g} days")
    return item("delete", du_bytes(ckpt), "D3.other_after_days", "not LATEST/best and older than window")


def _blocked_reason(run_dir: Path, run_id: str, status: str, lock: LockStatus | None) -> str | None:
    if (run_dir / "KEEP").exists():
        return "PINNED_OR_KEEP"
    if lock is not None and (lock.corrupt or lock.run_id is None or lock.run_id == run_id):
        return "GPU_LOCK_HOLDER"
    if status not in {s.value for s in RunStatus}:
        return "UNKNOWN_STATUS"  # missing/unreadable run.json or a status outside RunStatus
    if status not in CLEANABLE_STATUSES | FAILED_STATUSES:
        return "RUN_NOT_FINISHED"
    return None


def plan(
    root: Path,
    *,
    run_ids: Collection[str] | None = None,
    policy: RetentionPolicy = DEFAULT_POLICY,
    now: float | None = None,
    lock_reader: Callable[[Path], LockStatus | None] = check_lock,
    lock_path: Path | None = None,
) -> Plan:
    """Read-only retention plan for every run (or only ``run_ids``) under ``<root>/outputs/runs``.

    ``lock_path`` defaults to ``<root>/.team/GPU_LOCK``; the CLI passes ``gpu_lock.default_lock_path()``.
    """
    ts = time.time() if now is None else now
    lock = lock_reader(lock_path if lock_path is not None else root / ".team" / "GPU_LOCK")
    runs_dir = root / RUNS_REL
    runs: list[RunPlan] = []
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if run_dir.is_symlink() or not run_dir.is_dir() or (run_ids is not None and run_dir.name not in run_ids):
            continue
        status, ended_at = _run_status(run_dir)
        blocked = _blocked_reason(run_dir, run_dir.name, status, lock)
        failed_ended_at = None
        if status in FAILED_STATUSES:
            failed_ended_at = max(
                ended_at or 0.0, _newest_mtime(run_dir / "run.json"), _newest_mtime(run_dir / "checkpoints")
            )
        ckpt_root = run_dir / "checkpoints"
        items: list[PlanItem] = []
        if ckpt_root.is_dir() and not ckpt_root.is_symlink():
            roles = _roles(ckpt_root)
            for ckpt in sorted(ckpt_root.iterdir()):
                if ckpt.is_symlink() or not ckpt.is_dir() or not ckpt.name.startswith(CKPT_PREFIX):
                    continue
                items.append(
                    _plan_checkpoint(
                        run_dir.name,
                        ckpt,
                        roles.get(ckpt.name, []),
                        blocked=blocked,
                        failed_ended_at=failed_ended_at,
                        now=ts,
                        policy=policy,
                    )
                )
        runs.append(RunPlan(run_dir.name, status, (run_dir / "KEEP").exists(), tuple(items)))
    return Plan(ts=ts, runs=tuple(runs), policy=policy, lock=lock)


def _safe_target(root: Path, run_id: str, rel: str) -> Path:
    runs_dir = (root / RUNS_REL).resolve()
    target = root / RUNS_REL / run_id / rel
    if target.is_symlink():
        raise ValueError(f"refusing symlink: {target}")
    resolved = target.resolve()
    if runs_dir not in resolved.parents:
        raise ValueError(f"outside {runs_dir}: {resolved}")
    return target


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _check_project_root(root: Path) -> None:
    try:
        name = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{root} is not the {PROJECT_NAME} project root") from exc
    if name != PROJECT_NAME:
        raise ValueError(f"{root} is project {name!r}, expected {PROJECT_NAME!r}")


def _unapproved_bytes_since(log: Path, since: float) -> int:
    """Bytes deleted without ``approved_by`` after ``since`` (an unreadable line counts as 0, never as approved).

    This guards against accidentally splitting a large cleanup into small batches, not against someone editing or
    truncating ``cleanup.jsonl``; that is acceptable for the internal CLI. The app's delete path (P009, D17) uses an
    authenticated API with a server-side log instead.
    """
    if not log.exists():
        return 0
    total = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if not rec.get("approved_by") and float(rec.get("ts_epoch", 0)) >= since:
                total += int(rec.get("bytes", 0))
        except (ValueError, TypeError, AttributeError):
            continue
    return total


def apply(
    root: Path,
    item_ids: Iterable[str],
    *,
    by: str,
    approved_by: str | None = None,
    policy: RetentionPolicy = DEFAULT_POLICY,
    now: float | None = None,
    lock_reader: Callable[[Path], LockStatus | None] = check_lock,
    lock_path: Path | None = None,
) -> ApplyResult:
    """Execute the given planned items after re-planning; items whose plan changed are skipped.

    Raises:
        PermissionError: the selected items exceed ``approval_threshold_gb`` and no ``approved_by`` was given.
    """
    _check_project_root(root)
    ts = time.time() if now is None else now
    wanted = set(item_ids)
    fresh = plan(root, policy=policy, now=ts, lock_reader=lock_reader, lock_path=lock_path)
    by_id = {i.item_id: i for i in fresh.actionable}
    result = ApplyResult()
    selected = [by_id[i] for i in sorted(wanted) if i in by_id]
    result.skipped.extend((i, "no longer planned (state changed)") for i in sorted(wanted - by_id.keys()))
    if not approved_by:
        recent = _unapproved_bytes_since(root / CLEANUP_LOG_REL, ts - policy.approval_window_s)
        total_gb = (recent + sum(i.reclaim_bytes for i in selected)) / GIB
        if total_gb > policy.approval_threshold_gb:
            raise PermissionError(
                f"{total_gb:.1f} GiB (incl. {recent / GIB:.1f} GiB deleted without approval in the last "
                f"{policy.approval_window_s / 3600:g} h) > {policy.approval_threshold_gb:g} GiB needs --approved-by"
            )
    for item in selected:
        target = _safe_target(root, item.run_id, item.path)
        if item.action == "delete":
            shutil.rmtree(target)
        else:  # strip_optim
            for sub in STRIP_SUBDIRS:
                sub_path = target / sub
                if sub_path.is_dir() and not sub_path.is_symlink():
                    shutil.rmtree(sub_path)
        result.done.append(item)
        event: dict[str, object] = {
            "event": "checkpoint_pruned",
            "path": item.path,
            "action": item.action,
            "bytes": item.reclaim_bytes,
            "rule": item.rule,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts)),
            "ts_epoch": ts,
            "by": by,
            "approved_by": approved_by,
        }
        _append_jsonl(root / RUNS_REL / item.run_id / "events.jsonl", event)
        _append_jsonl(root / CLEANUP_LOG_REL, {**event, "run_id": item.run_id})
    return result


__all__ = [
    "DEFAULT_POLICY",
    "ApplyResult",
    "Plan",
    "PlanItem",
    "RetentionPolicy",
    "RunPlan",
    "apply",
    "plan",
]
