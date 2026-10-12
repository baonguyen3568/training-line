"""Storage panel API (P005 §6 R2, P009 §6): node-cache plan (dry-run) and user-chosen eviction. Owner: bao-cleanup.

- ``GET  /api/storage/plan``  read-only plan with a short-lived ``plan_id``; DISK-003 when the cap cannot be met
- ``POST /api/storage/apply`` ``{plan_id, item_ids, confirm?}``: re-plans, evicts only items of that plan that are
  still evictable; protected items (trained models, unreadable manifests) need ``confirm`` = their name; above the
  deployment's approval threshold answers 403 DISK-002 (an admin approves with the CLI, never from the UI)

There is no endpoint that deletes an arbitrary path. Text for the user is rendered here from i18n keys.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse

from vlm_pipeline import i18n
from vlm_pipeline.artifacts import cache_retention
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.common.diskguard import read_disk
from vlm_pipeline.common.paths import GIB

Locale = Annotated[str, Query(pattern="^(vi|en)$")]
ApprovalPolicy = Literal["none", "threshold"]
PLAN_TTL_S = 600.0
MAX_ITEMS = 500
POLICY_ENV = "VLM_STORAGE_APPROVAL_POLICY"
CAP_ENV = "VLM_NODE_CACHE_MAX_GB"
THRESHOLD_ENV = "VLM_STORAGE_APPROVAL_THRESHOLD_GB"


PROFILE_ENV = "VLM_PROFILE"
APPROVAL_WINDOW_S = 24 * 3600.0


@dataclass(frozen=True)
class StorageSettings:
    """``storage.*`` settings (ADR-005, P003 D17): approval policy per deployment profile, node-cache cap."""

    approval_policy: ApprovalPolicy = "none"
    approval_threshold_gb: float = 20.0
    node_cache_max_gb: float = cache_retention.DEFAULT_CAP_GB

    @classmethod
    def from_env(cls, root: Path) -> StorageSettings:
        """Profile default (``shared_dev`` -> threshold 20 GB, else none); env may only make it stricter.

        TODO(bao-backend): read the profile through the shared app/policy loader once it exists.
        """
        shared = os.environ.get(PROFILE_ENV, "personal") == "shared_dev"
        policy: ApprovalPolicy = "threshold" if shared else "none"
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
    """plan_id -> (created_at, item_ids) kept in memory for ``PLAN_TTL_S``."""

    clock: Callable[[], float] = time.time
    _plans: dict[str, tuple[float, frozenset[str]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, item_ids: frozenset[str]) -> tuple[str, float]:
        now = self.clock()
        plan_id = "pln_" + secrets.token_hex(8)
        with self._lock:
            self._plans = {k: v for k, v in self._plans.items() if now - v[0] < PLAN_TTL_S}
            self._plans[plan_id] = (now, item_ids)
        return plan_id, now + PLAN_TTL_S

    def get(self, plan_id: str) -> frozenset[str] | None:
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
    }


@dataclass
class StorageApi:
    root: Path
    index: ArtifactIndex
    settings: StorageSettings
    plans: _Plans = field(default_factory=_Plans)

    def plan(self, locale: str) -> dict[str, Any]:
        p = cache_retention.plan_cache(self.root, self.index, cap_gb=self.settings.node_cache_max_gb)
        auto = {i.cache_key for i in p.evict}
        items = [
            {**_item(item, reason), "auto": item.cache_key in auto}
            for item, reason in cache_retention.list_cache(self.root, self.index)
            if item.reclaim_bytes > 0
        ]
        plan_id, expires = self.plans.add(frozenset(i["item_id"] for i in items))
        threshold = self.settings.approval_threshold_gb if self.settings.approval_policy == "threshold" else None
        body: dict[str, Any] = {
            "schema_version": 2,
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
        by_key = {item.cache_key: (item, why) for item, why in cache_retention.list_cache(self.root, self.index)}
        chosen: list[str] = []
        confirmed: set[str] = set()
        for key in wanted:
            found = by_key.get(key)
            if found is None:
                skipped.append({"item_id": key, "reason": "NOT_EVICTABLE"})
                continue
            item, why = found
            if why is not None:
                if confirm.get(key) != item.run_id:
                    skipped.append({"item_id": key, "reason": "CONFIRM_REQUIRED"})
                    continue
                confirmed.add(key)
            chosen.append(key)
        chosen_bytes = sum(by_key[k][0].reclaim_bytes for k in chosen)
        recent = cache_retention.unapproved_manual_bytes_since(
            self.root / "outputs" / "system" / "cleanup.jsonl", time.time() - APPROVAL_WINDOW_S
        )
        total_gb = (chosen_bytes + recent) / GIB  # no splitting into small batches on shared hosts
        if self.settings.approval_policy == "threshold" and total_gb > self.settings.approval_threshold_gb:
            admin_command = (
                "uv run python scripts/cleanup/cache.py --items " + ",".join(chosen) + " --apply --by <admin-name>"
            )
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
                        admin_command=admin_command,
                    ),
                    "deleted": [],
                },
            )
        result = cache_retention.evict(self.root, self.index, chosen, by="app-user", manual=True, confirmed=confirmed)
        skipped += [{"item_id": k, "reason": why} for k, why in result.skipped]
        return JSONResponse(
            content={
                "evicted": [i.as_dict() for i in result.evicted],
                "skipped": skipped,
                "freed_gb": round(result.freed_bytes / GIB, 3),
            }
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
