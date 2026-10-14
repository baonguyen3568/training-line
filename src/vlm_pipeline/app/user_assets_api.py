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
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vlm_pipeline.app import user_assets
from vlm_pipeline.common.catalog import REPO_ID_PATTERN, set_user_model_lookup
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.download.resolver import RepoFilesInvalidError

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


class InspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(max_length=193, pattern=REPO_ID_PATTERN)
    revision: str = Field("main", max_length=128)
    config: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    split: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

    @field_validator("revision")
    @classmethod
    def _a_ref(cls, value: str) -> str:
        if not _REF.fullmatch(value):
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
    _register_dataset_routes(app, service, root)

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
        download = _hub_job(service, resolve_id, "model")
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


def _register_dataset_routes(app: FastAPI, service: Any, root: Path) -> None:
    """Inspect any Hugging Face dataset (P021 §3, H4): the resolve job, then the network-none inspect job."""
    inspections: dict[str, Path] = {}  # inspect_id -> the inspect job's out dir (thumbnails)

    @app.post("/api/user-assets/datasets/inspect", status_code=202)
    def inspect_dataset(body: InspectRequest) -> dict[str, Any]:
        from vlm_pipeline.app import user_datasets
        from vlm_pipeline.common.errors import VlmPipelineError
        from vlm_pipeline.common.jobrun import CONTAINER_ROOT

        _no_double_dash(body.repo)

        def container(path: Path) -> str:
            return f"{CONTAINER_ROOT}/{path.relative_to(root).as_posix()}"

        def finish(run_dir: Path) -> dict[str, Any]:
            meta = user_assets.read_resolve(run_dir)
            block = user_datasets.dataset_block(body.repo, meta.dataset)
            d = service.downloads[run_dir.name]
            name = f"{run_dir.name}-in"
            out = root / "outputs" / "runs" / name / "out"
            sample = run_dir / "stage" / "sample"
            if block.sample and sample.is_dir() and not sample.is_symlink():
                argv = ("python", "-m", "vlm_pipeline.download.inspect", "--sample", container(sample),
                        "--format", block.format, "--out", container(out))  # fmt: skip
                final = service.run_data_job(d, name, argv, (sample,))
                if final.status != RunStatus.COMPLETED:
                    raise RepoFilesInvalidError("inspect job failed")
                inspected = user_datasets.read_inspect(out)
            else:  # too big to sample, or no file the job could read: said, never guessed
                inspected = user_datasets.InspectResult(columns=[], suggested={}, samples=[], rows_read=0,
                                                        error="no_sample")  # fmt: skip
            inspections[run_dir.name] = out
            url = f"/api/user-assets/inspections/{run_dir.name}/thumbs"
            return user_datasets.api_result(body.repo, meta.sha, block, inspected, url)

        extra = {k: v for k, v in (("config", body.config), ("split", body.split)) if v}
        try:
            download = service.start_resolve("dataset", body.repo, body.revision, finish, extra)
        except VlmPipelineError as exc:  # busy: one hub job at a time
            raise HTTPException(status_code=409, detail={"code": exc.code, "params": exc.params}) from None
        return {"inspect_id": download.download_id}

    @app.get("/api/user-assets/inspections/{inspect_id}")
    def inspection(inspect_id: str) -> dict[str, Any]:
        download = _hub_job(service, inspect_id, "dataset")
        out: dict[str, Any] = {"status": _STATUS.get(download.status, "running")}
        if download.status == "completed":
            out["result"] = download.result
        if download.error is not None:
            out["error"] = download.error
        return out

    @app.get("/api/user-assets/inspections/{inspect_id}/thumbs/{n}.png")
    def thumb(inspect_id: str, n: int) -> FileResponse:
        from vlm_pipeline.app.user_datasets import thumb_path

        _hub_job(service, inspect_id, "dataset")
        folder = inspections.get(inspect_id)
        path = thumb_path(folder, n) if folder is not None and 0 <= n < 20 else None
        if path is None:
            raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store",
                                                                   "X-Content-Type-Options": "nosniff"})  # fmt: skip


def _hub_job(service: Any, job_id: str, kind: str) -> Any:
    download = service.downloads.get(job_id) if _RESOLVE_ID.fullmatch(job_id) else None
    if download is None or download.asset.catalog_id != "user-resolve" or download.asset.kind != kind:
        raise HTTPException(status_code=404, detail={"code": "CFG-004", "params": {}})
    return download


__all__ = ["REF_PATTERN", "register_user_asset_routes"]
