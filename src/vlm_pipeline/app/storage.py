"""Storage panel API (P005 §6 R2, P009 §6, P013 G-1): one dry-run plan over every cleanup source. Owner: bao-cleanup.

Sources (``item.source``): ``node_cache`` (cache_retention), ``checkpoint`` and ``set_aside`` (retention, ADR-005 D3).
Each item has a ``section``: ``auto`` (the scheduler may do it; regenerable or D3 without permanent loss), ``manual``
(only a person, with typed confirmation when protected or PERMANENT_LOSS) or ``kept`` (shown with its reason only).

- ``GET  /api/storage/plan``  read-only plan with a short-lived ``plan_id``; DISK-003 when the cache cap cannot be met
- ``POST /api/storage/apply`` ``{plan_id, item_ids, confirm?}``: re-plans, applies only items of that plan that are
  still applicable; protected / PERMANENT_LOSS items need ``confirm[item_id]`` = their run name; above the
  deployment's approval threshold (both sources and the last 24 h counted together) answers 403 DISK-002 (an admin
  approves with the CLI, never from the UI)

There is no endpoint that deletes an arbitrary path. Text for the user is rendered here from i18n keys.
"""

from __future__ import annotations

import os
import secrets
import shlex
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse

from vlm_pipeline import i18n
from vlm_pipeline.app.uploads import clean_label
from vlm_pipeline.artifacts import cache_retention, retention
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.artifacts.storage import du_bytes
from vlm_pipeline.common.app_config import effective_profile, read_policy
from vlm_pipeline.common.diskguard import read_disk
from vlm_pipeline.common.gpu_lock import PATH_ENV as GPU_LOCK_PATH_ENV
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import read_small_json

Locale = Annotated[str, Query(pattern="^(vi|en)$")]
ApprovalPolicy = Literal["none", "threshold"]
Source = Literal["node_cache", "checkpoint", "set_aside"]
CACHE: Source = "node_cache"
PERMANENT_LOSS = "PERMANENT_LOSS"
CLEANUP_LOG_REL = Path("outputs") / "system" / "cleanup.jsonl"
PLAN_TTL_S = 600.0
MAX_ITEMS = 500
POLICY_ENV = "VLM_STORAGE_APPROVAL_POLICY"
CAP_ENV = "VLM_NODE_CACHE_MAX_GB"
THRESHOLD_ENV = "VLM_STORAGE_APPROVAL_THRESHOLD_GB"


PROFILE_ENV = "VLM_PROFILE"
APPROVAL_WINDOW_S = 24 * 3600.0
RESUME_WARN_S = 24 * 3600.0  # PM: warn a day before an interrupted run stops being kept for a resume


@dataclass(frozen=True)
class StorageSettings:
    """``storage.*`` settings (ADR-005, P003 D17): approval policy per deployment profile, node-cache cap."""

    approval_policy: ApprovalPolicy = "none"
    approval_threshold_gb: float = 20.0
    node_cache_max_gb: float = cache_retention.DEFAULT_CAP_GB

    @classmethod
    def from_env(cls, root: Path, policy_file: Path | None = None) -> StorageSettings:
        """Profile default (``personal`` -> none, else threshold 20 GB); env may only make it stricter.

        The profile is ``effective_profile``: the admin policy file is the base and ``VLM_PROFILE`` may only tighten
        it. Fails closed (P013 N3): no profile anywhere counts as a shared host. ``policy_file`` defaults to the admin
        file (``app_config.policy_path()``).
        """
        profile = effective_profile(os.environ.get(PROFILE_ENV), read_policy(policy_file))
        policy: ApprovalPolicy = "none" if profile == "personal" else "threshold"
        env_policy = os.environ.get(POLICY_ENV)
        if env_policy is not None:
            if env_policy not in ("none", "threshold"):
                raise ValueError(f"{POLICY_ENV} must be none|threshold, got {env_policy!r}")
            if env_policy == "threshold":  # tighten only; "none" never relaxes a shared_dev host
                policy = "threshold"
        threshold = 20.0
        env_threshold = os.environ.get(THRESHOLD_ENV)
        if env_threshold is not None:
            threshold = min(threshold, float(env_threshold))  # lower only
        cap_env = os.environ.get(CAP_ENV)
        cap = float(cap_env) if cap_env else min(cache_retention.DEFAULT_CAP_GB, 0.1 * read_disk(root).free_gb)
        return cls(approval_policy=policy, approval_threshold_gb=threshold, node_cache_max_gb=cap)


@dataclass
class _Plans:
    """plan_id -> (created_at, {item_id: source}) kept in memory for ``PLAN_TTL_S``."""

    clock: Callable[[], float] = time.time
    _plans: dict[str, tuple[float, Mapping[str, Source]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, item_ids: Mapping[str, Source]) -> tuple[str, float]:
        now = self.clock()
        plan_id = "pln_" + secrets.token_hex(8)
        with self._lock:
            self._plans = {k: v for k, v in self._plans.items() if now - v[0] < PLAN_TTL_S}
            self._plans[plan_id] = (now, item_ids)
        return plan_id, now + PLAN_TTL_S

    def get(self, plan_id: str) -> Mapping[str, Source] | None:
        with self._lock:
            entry = self._plans.get(plan_id)
        if entry is None or self.clock() - entry[0] >= PLAN_TTL_S:
            return None
        return entry[1]


def _error(code: str, locale: str, **params: Any) -> dict[str, Any]:
    texts = i18n.load(locale)
    why = f"error.{code}.why"
    return {
        "code": code,
        "params": params,
        "title": i18n.t(f"error.{code}.title", locale, **params),
        "what": i18n.t(f"error.{code}.what", locale, **params),
        "why": i18n.t(why, locale, **params) if why in texts else None,
    }


def _item(item: cache_retention.CacheItem, protected_reason: str | None) -> dict[str, Any]:
    gb = round(item.reclaim_bytes / GIB, 3)
    if protected_reason is None:
        group, consequence = "temp_result", "WILL_RECOMPUTE"
    else:
        group, consequence = "trained_model", "PERMANENT_LOSS"
    return {
        "item_id": item.cache_key,
        "group": group,
        "display_name": None,
        "display_key": "storage.item.temp_result_of_run",
        "display_params": {"run_name": item.run_id},
        "size_gb": gb,
        "last_used": item.last_used,
        "run_id": item.run_id,
        "node_type": item.node_type,
        "action": "delete",
        "protected_reason": protected_reason,
        "regenerable": protected_reason is None,
        "consequence_code": consequence,
        "consequence_params": {"name": item.run_id} if consequence == "PERMANENT_LOSS" else {},
        "confirm_required": protected_reason is not None,
        "confirm_text": item.run_id if protected_reason is not None else None,
        "event_type": "cache_evicted",
        "source": CACHE,
    }


def _cache_entry(item: cache_retention.CacheItem, reason: str | None, auto: set[str], node: str = "") -> dict[str, Any]:
    """Panel row of a cache entry; one a later attempt resumes from (P017 C6) is only shown, never selectable."""
    if reason == retention.INPUT_REASON:
        row = _item(item, None)
        return {
            **row,
            "auto": False,
            "section": "kept",
            "consequence_code": None,
            "consequence_params": {},
            "lock_reason": reason,
            "lock_params": {"run": item.run_id, "node": node or item.run_id},
        }
    return {
        **_item(item, reason),
        "auto": item.cache_key in auto,
        "section": "auto" if item.cache_key in auto else "manual",
    }


def _retention_section(item: retention.PlanItem) -> str | None:
    """Auto / manual / kept for a retention item; None = nothing to show (an ordinary keep, e.g. still young)."""
    if item.action != "keep" and item.blocked_by is None:
        return "manual" if item.consequence_code == PERMANENT_LOSS else "auto"
    if item.action == "delete" and item.blocked_by == retention.CONFIRM_REASON:
        return "manual"
    return "kept" if item.blocked_by is not None else None


def _needs_confirm(item: retention.PlanItem) -> bool:
    return item.blocked_by == retention.CONFIRM_REASON or item.consequence_code == PERMANENT_LOSS


def _retention_item(
    item: retention.PlanItem, section: str, kept_bytes: int = 0, lock_params: dict[str, str] | None = None
) -> dict[str, Any]:
    """``kept_bytes``: on-disk size of a kept item, so the panel shows where space is held (KEEP, running...)."""
    ckpt = item.path.removeprefix("checkpoints/")
    confirm = section == "manual" and _needs_confirm(item)
    kept = section == "kept"
    return {
        "item_id": item.item_id,
        "source": item.kind if item.kind in ("checkpoint", "set_aside") else "checkpoint",
        "section": section,
        "auto": section == "auto",
        "group": "trained_model",
        "display_name": None,
        "display_key": "storage.item.set_aside_of_run"
        if item.kind == "set_aside"
        else "storage.item.checkpoint_of_run",
        "display_params": {"run_name": item.run_id, "checkpoint": ckpt},
        "size_gb": round((kept_bytes if kept else item.reclaim_bytes) / GIB, 3),
        "run_id": item.run_id,
        "action": item.action,
        "protected_reason": retention.CONFIRM_REASON if item.blocked_by == retention.CONFIRM_REASON else None,
        "lock_reason": item.blocked_by if kept else None,
        "lock_params": (lock_params or {"run": item.run_id}) if kept else {},
        "regenerable": False,
        "consequence_code": item.consequence_code,
        "consequence_params": dict(item.consequence_params),
        "confirm_required": confirm,
        "confirm_text": item.run_id if confirm else None,
        "event_type": "checkpoint_pruned",
    }


def _local_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _lock_params(root: Path, item: retention.PlanItem, locale: str) -> dict[str, str]:
    if item.blocked_by == retention.INTERRUPTED_REASON and item.resume_until is not None:
        return {"run": item.run_id, "until": _local_time(item.resume_until)}
    if item.blocked_by == retention.INPUT_REASON:
        return {"run": item.run_id, "node": _step_title(root, item, locale)}
    return {"run": item.run_id}


_UNTIL_FORMAT = {"vi": "%H:%M %d/%m", "en": "%b %d, %H:%M"}  # short date + time in a sentence (bao-design)


def _step_title(root: Path, item: retention.PlanItem, locale: str) -> str:
    """Node title of the run (app runs are named ``"<node type> <entry>"``), else the run's name, else its id."""
    return _title_of(root / item.runs_rel / item.run_id / "run.json", item.run_id, locale)


def _cache_title(root: Path, item: cache_retention.CacheItem, locale: str) -> str:
    for rel in cache_retention.RUN_ROOTS:
        run_json = root / rel / item.run_id / "run.json"
        if run_json.parent.is_dir() and not run_json.parent.is_symlink():
            return _title_of(run_json, item.run_id, locale)
    return item.run_id


def _title_of(run_json: Path, fallback: str, locale: str) -> str:
    try:  # a job can rewrite its own run.json: read safely, shown as a clean label (no bidi/control chars, short)
        data = read_small_json(run_json)
        name = clean_label(str(data.get("name") or "")) if isinstance(data, dict) else ""
    except (OSError, ValueError):
        name = ""
    key = f"node.{name.split(' ', 1)[0]}.title" if name else ""
    return i18n.t(key, locale) if key in i18n.load(locale) else (name or fallback)


def _resume_warnings(root: Path, items: list[retention.PlanItem], now: float, locale: str) -> list[dict[str, Any]]:
    """One notice per interrupted run whose resume window ends within ``RESUME_WARN_S`` (ADR-005, PM 2026-10-13)."""
    expiring = {
        i.run_id: i
        for i in items
        if i.blocked_by == retention.INTERRUPTED_REASON
        and i.resume_until is not None
        and 0 < i.resume_until - now <= RESUME_WARN_S
    }
    notices = []
    for run_id, item in sorted(expiring.items()):
        until = item.resume_until or now
        node = _step_title(root, item, locale)
        text_until = time.strftime(_UNTIL_FORMAT.get(locale, _UNTIL_FORMAT["en"]), time.localtime(until))
        notices.append(
            {
                "code": "RUN_INTERRUPTED_EXPIRING",
                "run_id": run_id,
                "node": node,
                "resume_until": _local_time(until),
                "message": i18n.t(
                    "storage.attention.RUN_INTERRUPTED_EXPIRING", locale, node=node, run=run_id, until=text_until
                ),
            }
        )
    return notices


@dataclass
class StorageApi:
    root: Path
    index: ArtifactIndex
    settings: StorageSettings
    plans: _Plans = field(default_factory=_Plans)
    lock_path: Path | None = None  # GPU lock file; default $VLM_GPU_LOCK_PATH or <root>/.team/GPU_LOCK

    def _lock_path(self) -> Path:
        if self.lock_path is not None:
            return self.lock_path
        override = os.environ.get(GPU_LOCK_PATH_ENV)
        return Path(override) if override else self.root / ".team" / "GPU_LOCK"

    def _kept_bytes(self, item: retention.PlanItem) -> int:
        """On-disk size of a kept item; 0 when its path is not a plain path inside its runs root."""
        try:
            return du_bytes(retention.item_path(self.root, item))
        except ValueError:
            return 0

    def _retention_items(self) -> list[tuple[retention.PlanItem, str]]:
        fresh = retention.plan(self.root, lock_path=self._lock_path())
        found = [(i, _retention_section(i)) for r in fresh.runs for i in r.items]
        return [(i, section) for i, section in found if section is not None]

    def plan(self, locale: str) -> dict[str, Any]:
        p = cache_retention.plan_cache(self.root, self.index, cap_gb=self.settings.node_cache_max_gb)
        auto = {i.cache_key for i in p.evict}
        items = [
            _cache_entry(
                item, reason, auto, _cache_title(self.root, item, locale) if reason == retention.INPUT_REASON else ""
            )
            for item, reason in cache_retention.list_cache(self.root, self.index)
            if item.reclaim_bytes > 0
        ]
        kept_or_actionable = self._retention_items()
        items += [
            _retention_item(
                item,
                section,
                self._kept_bytes(item) if section == "kept" else 0,
                _lock_params(self.root, item, locale) if section == "kept" else None,
            )
            for item, section in kept_or_actionable
        ]
        plan_id, expires = self.plans.add({i["item_id"]: i["source"] for i in items if i["section"] != "kept"})
        threshold = self.settings.approval_threshold_gb if self.settings.approval_policy == "threshold" else None
        body: dict[str, Any] = {
            "schema_version": 3,
            "plan_id": plan_id,
            "expires_at": expires,
            "mode": "dry_run",
            "approval_policy": self.settings.approval_policy,
            "threshold_gb": threshold,
            "cap_gb": round(self.settings.node_cache_max_gb, 1),
            "cache_gb": round(p.cache_bytes / GIB, 3),
            "auto_reclaim_gb": round(p.reclaim_bytes / GIB, 3),
            "shortfall_gb": round(p.shortfall_bytes / GIB, 3),
            "items": items,
            "attention": _resume_warnings(self.root, [i for i, _ in kept_or_actionable], time.time(), locale),
            "error": None,
        }
        if p.shortfall_bytes > 0:
            body["error"] = _error(
                "DISK-003",
                locale,
                gb_over=round(p.shortfall_bytes / GIB, 1),
                cap_gb=round(self.settings.node_cache_max_gb, 1),
                protected_count=len(p.protected),
            )
        return body

    def apply(self, plan_id: str, item_ids: list[str], confirm: Mapping[str, str], locale: str) -> JSONResponse:
        planned = self.plans.get(plan_id)
        if planned is None:
            return JSONResponse(status_code=409, content={"code": "PLAN_EXPIRED", "plan_id": plan_id})
        wanted = [i for i in dict.fromkeys(item_ids) if i in planned][:MAX_ITEMS]
        skipped: list[dict[str, str]] = [{"item_id": i, "reason": "NOT_IN_PLAN"} for i in item_ids if i not in planned]
        cache_keys, cache_confirmed, cache_bytes = self._choose_cache(
            [k for k in wanted if planned[k] == CACHE], confirm, skipped
        )
        pruned, pruned_confirmed = self._choose_retention([k for k in wanted if planned[k] != CACHE], confirm, skipped)
        log = self.root / CLEANUP_LOG_REL
        since = time.time() - APPROVAL_WINDOW_S
        recent = cache_retention.unapproved_manual_bytes_since(log, since) + retention.unapproved_bytes_since(
            log, since
        )
        total_gb = (cache_bytes + sum(i.reclaim_bytes for i in pruned) + recent) / GIB  # no splitting into batches
        if self.settings.approval_policy == "threshold" and total_gb > self.settings.approval_threshold_gb:
            return self._approval_required(total_gb, cache_keys, pruned, pruned_confirmed, locale)
        with retention.cleanup_lock(self.root) as held:
            if not held:  # the scheduler or a CLI pass is deleting right now; the user retries in a moment
                return JSONResponse(status_code=409, content={"code": "CLEANUP_IN_PROGRESS", "plan_id": plan_id})
            return self._apply_chosen(cache_keys, cache_confirmed, pruned, pruned_confirmed, skipped)

    def _apply_chosen(
        self,
        cache_keys: list[str],
        cache_confirmed: set[str],
        pruned: list[retention.PlanItem],
        pruned_confirmed: set[str],
        skipped: list[dict[str, str]],
    ) -> JSONResponse:
        result = cache_retention.evict(
            self.root, self.index, cache_keys, by="app-user", manual=True, confirmed=cache_confirmed
        )
        skipped += [{"item_id": k, "reason": why} for k, why in result.skipped]
        done = retention.apply(
            self.root,
            [i.item_id for i in pruned],
            by="app-user",
            manual=True,
            confirmed=pruned_confirmed,
            policy=retention.RetentionPolicy(approval_threshold_gb=float("inf")),  # checked above for both sources
            lock_path=self._lock_path(),
        )
        skipped += [{"item_id": k, "reason": why} for k, why in done.skipped]
        return JSONResponse(
            content={
                "evicted": [i.as_dict() for i in result.evicted],
                "pruned": [i.as_dict() for i in done.done],
                "skipped": skipped,
                "freed_gb": round((result.freed_bytes + done.freed_bytes) / GIB, 3),
            }
        )

    def _choose_cache(
        self, keys: list[str], confirm: Mapping[str, str], skipped: list[dict[str, str]]
    ) -> tuple[list[str], set[str], int]:
        by_key = {item.cache_key: (item, why) for item, why in cache_retention.list_cache(self.root, self.index)}
        chosen: list[str] = []
        confirmed: set[str] = set()
        for key in keys:
            found = by_key.get(key)
            if found is None:
                skipped.append({"item_id": key, "reason": "NOT_EVICTABLE"})
                continue
            item, why = found
            if why == retention.INPUT_REASON:  # a later attempt resumes from it: no confirmation unlocks it
                skipped.append({"item_id": key, "reason": why})
                continue
            if why is not None:
                if confirm.get(key) != item.run_id:
                    skipped.append({"item_id": key, "reason": "CONFIRM_REQUIRED"})
                    continue
                confirmed.add(key)
            chosen.append(key)
        return chosen, confirmed, sum(by_key[k][0].reclaim_bytes for k in chosen)

    def _choose_retention(
        self, keys: list[str], confirm: Mapping[str, str], skipped: list[dict[str, str]]
    ) -> tuple[list[retention.PlanItem], set[str]]:
        by_id = {i.item_id: (i, section) for i, section in self._retention_items()}
        chosen: list[retention.PlanItem] = []
        confirmed: set[str] = set()
        for key in keys:
            found = by_id.get(key)
            if found is None or found[1] == "kept":
                reason = found[0].blocked_by if found is not None else None
                skipped.append({"item_id": key, "reason": reason or "NOT_APPLICABLE"})
                continue
            item = found[0]
            if _needs_confirm(item):
                if confirm.get(key) != item.run_id:
                    skipped.append({"item_id": key, "reason": "CONFIRM_REQUIRED"})
                    continue
                if item.blocked_by == retention.CONFIRM_REASON:
                    confirmed.add(key)
            chosen.append(item)
        return chosen, confirmed

    def _approval_required(
        self,
        total_gb: float,
        cache_keys: list[str],
        pruned: list[retention.PlanItem],
        pruned_confirmed: set[str],
        locale: str,
    ) -> JSONResponse:
        commands = []  # ids are directory names an admin copy-pastes into a shell: always shlex.quote them
        if cache_keys:
            commands.append(
                "uv run python scripts/cleanup/cache.py --items "
                + shlex.quote(",".join(cache_keys))
                + " --apply --by <admin-name>"
            )
        if pruned:
            confirm = f" --confirm {shlex.quote(','.join(sorted(pruned_confirmed)))}" if pruned_confirmed else ""
            commands.append(
                "uv run python scripts/cleanup/retention.py --items "
                + shlex.quote(",".join(i.item_id for i in pruned))
                + confirm
                + " --apply --by <admin-name> --approved-by <admin-name>"
            )
        chosen = cache_keys + [i.item_id for i in pruned]
        return JSONResponse(
            status_code=403,
            content={
                "error": _error(
                    "DISK-002",
                    locale,
                    gb=round(total_gb, 1),
                    threshold_gb=self.settings.approval_threshold_gb,
                    item_count=len(chosen),
                    item_ids=chosen,
                    admin_command=" && ".join(commands),
                ),
                "deleted": [],
            },
        )


def on_startup(index: ArtifactIndex, active_graph_run_ids: list[str]) -> int:
    """Release cache leases of graph runs that died with a previous app process."""
    return index.release_all_except(active_graph_run_ids)


def register_storage_routes(app: FastAPI, api: StorageApi) -> None:
    @app.get("/api/storage/plan")
    def storage_plan(locale: Locale = i18n.DEFAULT_LOCALE) -> dict[str, Any]:
        return api.plan(locale)

    @app.post("/api/storage/apply")
    def storage_apply(
        plan_id: Annotated[str, Body(max_length=64)],
        item_ids: Annotated[list[str], Body(max_length=MAX_ITEMS)],
        confirm: Annotated[dict[str, str] | None, Body()] = None,
        locale: Locale = i18n.DEFAULT_LOCALE,
    ) -> JSONResponse:
        return api.apply(plan_id, item_ids, confirm or {}, locale)
