"""HTTP API of user-added models (ADR-021, P021 §8): add (async resolve), list, confirm the licence, delete.

* ``POST /api/user-assets/models {repo, revision?}`` -> 202 ``{resolve_id}``: the plan-only hub job runs in the one
  download slot (busy -> 409 with the busy error), then the host resolver builds the record (saved, licence not
  confirmed). A malformed repo id or ref -> 400 at once, before any job.
* ``GET /api/user-assets/resolves/{resolve_id}`` -> ``{status, model?, error?}``; ``GET /api/user-assets/models``;
  ``POST /api/user-assets/models/{id}/confirm-license {accept: true}``; ``DELETE /api/user-assets/models/{id}``
  (the record only: downloaded files go with the storage cleanup).
Behind ``SecurityMiddleware`` like every route (session, Host/Origin, CSRF on writes).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vlm_pipeline.app import user_assets
from vlm_pipeline.common.catalog import REPO_ID_PATTERN, set_user_model_lookup

# A branch, tag or commit; never an unmerged hub pull request (security H2 R2). The job checks it again.
REF_PATTERN = r"^(?!.*\.\.)(?!refs/pr/)[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"
_REF = re.compile(REF_PATTERN)
_RESOLVE_ID = re.compile(r"^[0-9]{8}-[0-9]{4}-rs-[0-9a-f]{8}$")
_STATUS = {"queued": "queued", "downloading": "running", "verifying": "running", "completed": "done",
           "failed": "failed"}  # fmt: skip


class AddModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(max_length=193, pattern=REPO_ID_PATTERN)
    revision: str = Field("main", max_length=128)

    @field_validator("revision")
    @classmethod
    def _a_ref(cls, value: str) -> str:
        if not _REF.fullmatch(value):  # Python re: pydantic's pattern engine has no lookahead
            raise ValueError("not a branch, tag or commit of the repo")
        return value


class ConfirmLicenseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accept: Literal[True]


def _no_double_dash(repo: str) -> None:
    if "--" in repo or ".." in repo:  # security checklist §2
        raise HTTPException(status_code=400, detail={"code": "CFG-005", "params": {"param": "repo", "rule": "format"}})


def register_user_asset_routes(app: FastAPI) -> None:
    service = getattr(app.state, "downloads", None)
    if service is None:
        return
    root = service.root

    def lookup(model_id: str) -> Any:
        record = user_assets.load(root, model_id)
        return record.entry if record is not None else None

    set_user_model_lookup(lookup)  # common.catalog.model_entry finds user-added models from now on

    @app.post("/api/user-assets/models", status_code=202)
    def add_model(body: AddModelRequest) -> dict[str, Any]:
        from vlm_pipeline.common.errors import VlmPipelineError

        _no_double_dash(body.repo)

        def finish(run_dir: Any) -> dict[str, Any]:
            record = user_assets.finish_model(root, run_dir, body.repo)
            kept = user_assets.load(root, record.entry.id)  # re-adding keeps the licence answer and the trial
            if kept is not None:
                record = record.model_copy(update={"license_confirmed": kept.license_confirmed,
                                                   "license_confirmed_at": kept.license_confirmed_at,
                                                   "license_confirmed_sha": kept.license_confirmed_sha,
                                                   "trial_run_id": kept.trial_run_id})  # fmt: skip
            sha = user_assets.save_template(root, record)
            record = record.model_copy(update={"template_sha256": sha})
            user_assets.save(root, record)
            return record.to_api()

        try:
            download = service.start_resolve("model", body.repo, body.revision, finish)
        except VlmPipelineError as exc:  # busy: one hub job at a time
            raise HTTPException(status_code=409, detail={"code": exc.code, "params": exc.params}) from None
        return {"resolve_id": download.download_id}

    @app.get("/api/user-assets/resolves/{resolve_id}")
    def resolve_status(resolve_id: str) -> dict[str, Any]:
        download = service.downloads.get(resolve_id) if _RESOLVE_ID.fullmatch(resolve_id) else None
        if download is None or download.asset.catalog_id != "user-resolve":
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        out: dict[str, Any] = {"status": _STATUS.get(download.status, "running")}
        if download.status == "completed":
            out["model"] = download.result
        if download.error is not None:
            out["error"] = download.error
        return out

    @app.get("/api/user-assets/models")
    def list_models() -> dict[str, Any]:
        return {"models": [r.to_api() for r in user_assets.list_all(root)]}

    @app.post("/api/user-assets/models/{model_id}/confirm-license")
    def confirm(model_id: str, body: ConfirmLicenseRequest) -> dict[str, Any]:
        del body  # accept is Literal[True]: the request itself is the confirmation
        record = user_assets.confirm_license(root, model_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return {"model": record.to_api()}

    @app.delete("/api/user-assets/models/{model_id}")
    def remove(model_id: str) -> dict[str, Any]:
        if not user_assets.delete(root, model_id):
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return {"deleted": model_id}


__all__ = ["REF_PATTERN", "register_user_asset_routes"]
