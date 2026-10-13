"""Storage panel: "Reset this step" (G-4, PM 2026-10-13). Owner: bao-cleanup.

- ``GET  /api/storage/node-reset?node=<nid>-<key16>``: read-only plan of ``node_reset.plan_reset`` with a short-lived
  ``plan_id`` (None when refused or there is nothing to delete); ``blocked_by`` uses the ``storage.lock_reason`` keys
- ``POST /api/storage/node-reset`` ``{plan_id, node, confirm?}``: deletes every attempt dir of that node and bumps its
  generation, so the executor gives it new run dir names (P017 claims are never reused). Checkpoints make it a
  PERMANENT_LOSS: ``confirm`` must match the node's title (bao-design: a person types "Dạy model", never a dir
  name; compared case- and accent-insensitively, see ``same_text``). Counts against the shared-host approval
  threshold together with the other manual deletes (403 DISK-002 with the admin CLI command); 409 ``PLAN_EXPIRED``,
  ``CLEANUP_IN_PROGRESS`` or the refusal reason when the step became busy since the dry-run.
"""

from __future__ import annotations

import secrets
import shlex
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Body, FastAPI, Query
from fastapi.responses import JSONResponse

from vlm_pipeline import i18n
from vlm_pipeline.artifacts import node_reset, retention
from vlm_pipeline.common.paths import GIB

if TYPE_CHECKING:
    from vlm_pipeline.app.storage import StorageApi

PLAN_TTL_S = 600.0
NODE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}-[0-9a-f]{16}$"  # run_claims.NODE_BASE


def same_text(typed: str | None, expected: str) -> bool:
    """Typed confirmation: equal after NFC, dropping accents (đ -> d), case folding and collapsing spaces."""

    def norm(text: str) -> str:
        decomposed = unicodedata.normalize(
            "NFD", unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")
        )
        bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return " ".join(bare.casefold().split())

    return typed is not None and bool(expected.strip()) and norm(typed) == norm(expected)


@dataclass
class ResetPlans:
    """plan_id -> (created_at, plan) for ``PLAN_TTL_S``: an apply only executes what the user looked at."""

    _plans: dict[str, tuple[float, node_reset.ResetPlan, str]] = field(default_factory=dict)  # (created, plan, title)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, plan: node_reset.ResetPlan, title: str, now: float) -> tuple[str, float]:
        plan_id = "rst_" + secrets.token_hex(8)
        with self._lock:
            self._plans = {k: v for k, v in self._plans.items() if now - v[0] < PLAN_TTL_S}
            self._plans[plan_id] = (now, plan, title)
        return plan_id, now + PLAN_TTL_S

    def drop(self, plan_id: str) -> None:
        with self._lock:
            self._plans.pop(plan_id, None)

    def get(self, plan_id: str, now: float) -> tuple[node_reset.ResetPlan, str] | None:
        with self._lock:
            entry = self._plans.get(plan_id)
        return (entry[1], entry[2]) if entry is not None and now - entry[0] < PLAN_TTL_S else None


def reset_plan(api: StorageApi, plans: ResetPlans, node: str, locale: str) -> dict[str, Any]:
    from vlm_pipeline.app.storage import title_of

    plan = node_reset.plan_reset(api.root, api.index, node, lock_path=api.lock_file())
    runs = api.root / retention.GRAPH_RUNS_REL
    title = title_of(runs / plan.attempts[0].name / "run.json", node, locale) if plan.attempts else node
    usable = plan.blocked_by is None and bool(plan.attempts)
    plan_id, expires = plans.add(plan, title, api.plans.clock()) if usable else (None, None)
    confirm = plan.consequence_code == node_reset.PERMANENT_LOSS
    return {
        "schema_version": 1,
        "plan_id": plan_id,
        "expires_at": expires,
        "node": node,
        "node_title": title,
        "attempts": [
            {
                "run_dir": a.name,
                "status": a.status,
                "size_gb": round(a.reclaim_bytes / GIB, 3),
                "has_checkpoint": a.has_checkpoint,
            }
            for a in plan.attempts
        ],
        "total_gb": round(plan.reclaim_bytes / GIB, 3),
        "blocked_by": plan.blocked_by,
        "blocked_params": {**plan.blocked_params, "node": title} if plan.blocked_by else {},
        "consequence_code": plan.consequence_code,
        "consequence_params": {"name": title} if confirm else {},
        "confirm_required": confirm,
        "confirm_text": title if confirm else None,
        "next_generation": plan.generation + 1,
    }


def reset_apply(
    api: StorageApi, plans: ResetPlans, plan_id: str, node: str, *, confirm: str | None, locale: str
) -> JSONResponse:
    found = plans.get(plan_id, api.plans.clock())
    if found is None or found[0].node != node:
        return JSONResponse(status_code=409, content={"code": "PLAN_EXPIRED", "plan_id": plan_id})
    expected, title = found
    if expected.consequence_code == node_reset.PERMANENT_LOSS and not same_text(confirm, title):
        return JSONResponse(status_code=400, content={"code": "CONFIRM_REQUIRED", "confirm_text": title})
    total_gb = (expected.reclaim_bytes + api.recent_unapproved_bytes()) / GIB
    if api.settings.approval_policy == "threshold" and total_gb > api.settings.approval_threshold_gb:
        confirm_arg = f" --confirm {shlex.quote(node)}" if expected.consequence_code else ""
        command = (
            f"uv run python scripts/cleanup/node_reset.py --node {shlex.quote(node)}{confirm_arg}"
            " --apply --by <admin-name> --approved-by <admin-name>"
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": api.error(
                    "DISK-002",
                    locale,
                    gb=round(total_gb, 1),
                    threshold_gb=api.settings.approval_threshold_gb,
                    item_count=len(expected.attempts),
                    item_ids=[node],
                    admin_command=command,
                ),
                "deleted": [],
            },
        )
    with retention.cleanup_lock(api.root) as held:
        if not held:
            return JSONResponse(status_code=409, content={"code": "CLEANUP_IN_PROGRESS", "plan_id": plan_id})
        try:
            result = node_reset.apply_reset(
                api.root, api.index, node, by="app-user", expected=expected, lock_path=api.lock_file()
            )
        except PermissionError as exc:
            return JSONResponse(status_code=409, content={"code": str(exc), "plan_id": plan_id})
    plans.drop(plan_id)  # one reset per look at the plan
    return JSONResponse(
        status_code=200,
        content={
            "deleted": result.deleted,
            "failed": result.failed,
            "freed_gb": round(result.freed_bytes / GIB, 3),
            "generation": result.generation,
        },
    )


def register_reset_routes(app: FastAPI, api: StorageApi) -> None:
    plans = ResetPlans()

    @app.get("/api/storage/node-reset")
    def storage_node_reset_plan(
        node: Annotated[str, Query(pattern=NODE_PATTERN)],
        locale: Annotated[str, Query(pattern="^(vi|en)$")] = i18n.DEFAULT_LOCALE,
    ) -> dict[str, Any]:
        return reset_plan(api, plans, node, locale)

    @app.post("/api/storage/node-reset")
    def storage_node_reset_apply(
        plan_id: Annotated[str, Body(max_length=64)],
        node: Annotated[str, Body(pattern=NODE_PATTERN)],
        confirm: Annotated[str | None, Body(max_length=128)] = None,
        locale: Annotated[str, Query(pattern="^(vi|en)$")] = i18n.DEFAULT_LOCALE,
    ) -> JSONResponse:
        return reset_apply(api, plans, plan_id, node, confirm=confirm, locale=locale)


__all__ = ["register_reset_routes", "reset_apply", "reset_plan"]
