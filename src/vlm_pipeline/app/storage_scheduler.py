"""Scheduled cleanup inside the app (P013 G-2, PM decisions 2026-10-13). Owner: bao-cleanup.

Every ``interval_s`` (first run ``first_delay_s`` after start) one pass:

1. skips when another pass holds ``outputs/system/retention.lock`` (CLI, a second app) or when the GPU lock file exists
   at all (busy, stale or unreadable: no IO/RAM contention with a training job, simpler safety reasoning);
2. picks only *safe* items: node-cache entries ``plan_cache`` chooses (regenerable, never protected) and retention
   items that are actionable and not ``PERMANENT_LOSS``. Blocked, protected and permanent-loss items are left to a
   person in the storage panel;
3. on a ``threshold`` deployment keeps retention deletes within the unapproved 24 h quota (it never approves itself);
   what does not fit waits for a later pass. Cache evictions of regenerable results are not counted (ADR-005: the
   cache cap is enforced automatically by design);
4. ``mode``: ``apply`` deletes (``by="app-scheduler"``); ``dry_run`` only writes the pass to
   ``outputs/system/retention_plan.json``; ``off`` does nothing. See ``mode_from_env``: the admin policy file
   (``~/.config/vlm-app/policy.toml``, ``[storage] schedule``) or else the profile default decides, and
   ``VLM_RETENTION_SCHEDULE`` may only lower it.

The pass holds ``retention.cleanup_lock`` (shared with ``scripts/cleanup/{retention,cache}.py --apply``).
"""

from __future__ import annotations

import json
import logging
import os
import pwd
import stat
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vlm_pipeline.app.storage import APPROVAL_WINDOW_S, CLEANUP_LOG_REL, PERMANENT_LOSS, PROFILE_ENV, StorageSettings
from vlm_pipeline.artifacts import cache_retention, retention
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.common.paths import GIB

logger = logging.getLogger(__name__)

Mode = Literal["off", "dry_run", "apply"]
MODE_ENV = "VLM_RETENTION_SCHEDULE"
BY = "app-scheduler"
LAST_PASS_REL = Path("outputs") / "system" / "retention_plan.json"
FIRST_DELAY_S = 600.0
INTERVAL_S = 6 * 3600.0


_RANK: dict[str, int] = {"off": 0, "dry_run": 1, "apply": 2}
POLICY_REL = Path(".config") / "vlm-app" / "policy.toml"  # admin-owned (P003 D17), never written by the app


def policy_path() -> Path:
    """``~/.config/vlm-app/policy.toml`` of the *process user* from the passwd database, never from ``$HOME``."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / POLICY_REL


def _unsafe(st: os.stat_result) -> bool:
    return bool(st.st_mode & 0o022) or st.st_uid not in (0, os.getuid())


def _read_policy(path: Path) -> dict[str, Any] | None:
    """The parsed policy file, or None when absent or unusable (reason logged). Checks the opened fd itself."""
    try:
        if _unsafe(path.parent.stat()):  # others could swap the file by rename
            logger.warning("policy dir %s is group/other-writable or foreign-owned: ignored", path.parent)
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP for a symlink
        logger.warning("policy file %s unusable (%s): ignored", path, exc)
        return None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if _unsafe(st) or not stat.S_ISREG(st.st_mode):
            logger.warning("policy file %s is not a regular file owned by this user/root and 0644: ignored", path)
            return None
        try:
            return tomllib.load(fh)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            logger.warning("policy file %s invalid (%s): ignored", path, exc)
            return None


def admin_schedule(path: Path | None = None) -> Mode | None:
    """``[storage] schedule`` from the admin policy file, or None (absent / unusable -> no admin decision).

    Interim reader for this one key until the shared ``app/policy.py`` loader (D17) exists; other keys are ignored.
    Fails closed: a symlink, a file or parent dir writable by group/others or not owned by this user or root, an
    unreadable file or an unknown value is ignored (logged), leaving the profile default (dry_run on a shared host).
    """
    data = _read_policy(path if path is not None else policy_path())
    storage = data.get("storage") if data is not None else None
    value = storage.get("schedule") if isinstance(storage, dict) else None
    if value is None:
        return None
    if value not in _RANK:
        logger.warning("storage.schedule must be off|dry_run|apply, got %r: ignored", value)
        return None
    return value  # type: ignore[no-any-return]


def mode_from_env(policy: Path | None = None) -> Mode:
    """Effective mode: admin policy (or the profile default), then only tightened by ``VLM_RETENTION_SCHEDULE``.

    Profile default fails closed: ``apply`` only for an explicit ``VLM_PROFILE=personal``; ``shared_dev`` or unset ->
    ``dry_run``. On a shared host only the admin file (``[storage] schedule = "apply"``) turns deleting on; env can
    lower the mode (``off``/``dry_run``) but never raise it (D17).
    """
    default: Mode = "apply" if os.environ.get(PROFILE_ENV) == "personal" else "dry_run"
    base = admin_schedule(policy) or default
    value = os.environ.get(MODE_ENV)
    if value is None:
        return base
    if value not in _RANK:
        raise ValueError(f"{MODE_ENV} must be off|dry_run|apply, got {value!r}")
    if _RANK[value] > _RANK[base]:
        logger.warning("%s=%s ignored: env may only lower the mode (%s)", MODE_ENV, value, base)
        return base
    return value  # type: ignore[return-value]


@dataclass
class RetentionScheduler:
    root: Path
    index: ArtifactIndex
    settings: StorageSettings
    lock_path: Path  # GPU lock file
    mode: Mode = "dry_run"
    first_delay_s: float = FIRST_DELAY_S
    interval_s: float = INTERVAL_S
    clock: Callable[[], float] = time.time
    _stop: threading.Event = field(default_factory=threading.Event)

    def run_once(self) -> dict[str, Any]:
        """One pass; returns (and records) its summary. Never raises for a skipped pass."""
        if self.mode == "off":
            return {"status": "off"}
        with retention.cleanup_lock(self.root) as held:
            if not held:
                return self._record({"status": "skipped", "reason": "PASS_IN_PROGRESS"})
            return self._record(self._pass())

    def _pass(self) -> dict[str, Any]:
        if self.lock_path.exists() or self.lock_path.is_symlink():
            return {"status": "skipped", "reason": "GPU_LOCK_HELD"}
        cache_plan = cache_retention.plan_cache(self.root, self.index, cap_gb=self.settings.node_cache_max_gb)
        cache_keys = [i.cache_key for i in cache_plan.evict]
        ret_plan = retention.plan(self.root, lock_path=self.lock_path)
        safe = [i for i in ret_plan.actionable if i.consequence_code != PERMANENT_LOSS]
        chosen, deferred = self._within_quota(safe)
        summary: dict[str, Any] = {
            "status": "dry_run" if self.mode == "dry_run" else "applied",
            "cache_evict": cache_keys,
            "cache_shortfall_gb": round(cache_plan.shortfall_bytes / GIB, 3),
            "retention": [i.item_id for i in chosen],
            "retention_deferred_by_quota": [i.item_id for i in deferred],
            "left_to_a_person": [
                i.item_id
                for r in ret_plan.runs
                for i in r.items
                if i.action != "keep" and i not in safe and (i.blocked_by in (None, retention.CONFIRM_REASON))
            ],
        }
        if self.mode != "apply":
            return summary
        evicted = cache_retention.evict(self.root, self.index, cache_keys, by=BY)
        done = retention.apply(
            self.root,
            [i.item_id for i in chosen],
            by=BY,
            policy=self._policy(),
            lock_path=self.lock_path,
        )
        summary["freed_gb"] = round((evicted.freed_bytes + done.freed_bytes) / GIB, 3)
        summary["skipped"] = [k for k, _ in evicted.skipped] + [k for k, _ in done.skipped]
        return summary

    def _policy(self) -> retention.RetentionPolicy:
        if self.settings.approval_policy == "threshold":
            return retention.RetentionPolicy(approval_threshold_gb=self.settings.approval_threshold_gb)
        return retention.RetentionPolicy(approval_threshold_gb=float("inf"))

    def _within_quota(
        self, items: list[retention.PlanItem]
    ) -> tuple[list[retention.PlanItem], list[retention.PlanItem]]:
        if self.settings.approval_policy != "threshold":
            return items, []
        since = self.clock() - APPROVAL_WINDOW_S
        room = self.settings.approval_threshold_gb * GIB - retention.unapproved_bytes_since(
            self.root / CLEANUP_LOG_REL, since
        )
        chosen: list[retention.PlanItem] = []
        deferred: list[retention.PlanItem] = []
        for item in items:
            if item.reclaim_bytes <= room:
                chosen.append(item)
                room -= item.reclaim_bytes
            else:
                deferred.append(item)
        return chosen, deferred

    def _record(self, summary: dict[str, Any]) -> dict[str, Any]:
        summary = {"ts": self.clock(), "mode": self.mode, **summary}
        path = self.root / LAST_PASS_REL
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)  # never write through a planted symlink
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False, indent=1))
            tmp.replace(path)
        except OSError as exc:
            logger.warning("cannot write %s: %s", path, exc)
        return summary

    def _loop(self) -> None:
        delay = self.first_delay_s
        while not self._stop.wait(delay):
            try:
                summary = self.run_once()
                logger.info("retention pass: %s", summary.get("status"))
            except Exception:  # a failing pass must not kill the app; the next one retries
                logger.exception("retention pass failed")
            delay = self.interval_s

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._loop, name="retention-scheduler", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
