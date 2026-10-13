"""Catalog downloads started from the app (P018 / G-5): budget, confirmation, one at a time, progress, install.

``DownloadService.start(kind, catalog_id)`` resolves EVERYTHING from the catalog (repo, 40-hex revision, size,
licence, pinned remote-code files); a client only names a catalog id. Before any job:

* already installed -> done at once (the snapshot is re-checked when the next download of it lands);
* budget (security download-egress §5): ``planned = size * 1.05 + 100 MB`` must fit the admin cap
  (``[download] max_gb``, NET-004) and the free disk minus the reserve (50 GB / 10 % on ``shared_dev``, 20 GB on
  ``personal``; DISK-001);
* above ``CONFIRM_GB`` the caller must confirm (``needs_confirm``) before anything starts;
* ONE download at a time (``outputs/system/downloads/active``, O_EXCL): it holds no GPU lock.

The job runs through vlm-runner (``RunnerLauncher``: egress proxy + token file); the host then verifies and
installs (``download.install``). Progress, verifying, installing, completed / failed reach the EventBus under the
download id, like a run's events.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from vlm_pipeline.common.app_config import effective_profile, read_policy
from vlm_pipeline.common.catalog import load_catalog, load_dataset_catalog
from vlm_pipeline.common.diskguard import read_disk
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.jobrun import CONTAINER_ROOT, ContainerJob, run_job
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.run_claims import holder_gone, process_identity
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_small_json
from vlm_pipeline.download.hub import SPEC_NAME
from vlm_pipeline.download.install import (
    DownloadIntegrityError,
    install,
    pinned_hashes,
    snapshot_dir,
    verified,
    verify_existing,
)

logger = logging.getLogger(__name__)

Kind = Literal["model", "dataset"]
CONFIRM_GB = 20.0
ACTIVE_REL = Path("outputs") / "system" / "downloads" / "active"
_ID_PART = re.compile(r"[^a-z0-9-]+")


class DownloadQuotaError(VlmPipelineError):
    code: ClassVar[str] = "NET-004"


class DownloadDiskError(VlmPipelineError):
    code: ClassVar[str] = "DISK-001"


class DownloadUnknownError(VlmPipelineError):
    """Not a catalog id of that kind (or a dataset security rejected): nothing is downloaded."""

    code: ClassVar[str] = "CFG-004"


class DownloadBusyError(VlmPipelineError):
    """Another download is running (one at a time); the UI retries when it ends."""

    code: ClassVar[str] = "SYS-002"


@dataclass(frozen=True)
class Asset:
    kind: Kind
    catalog_id: str
    repo: str
    revision: str
    name: str
    size_gb: float
    license: str
    gated: bool

    def to_api(self) -> dict[str, Any]:
        return {"kind": self.kind, "catalog_id": self.catalog_id, "name": self.name, "size_gb": self.size_gb,
                "license": self.license, "gated": self.gated}  # fmt: skip


def resolve(kind: Kind, catalog_id: str) -> Asset:
    """The catalog entry behind ``catalog_id``; a client never sends a repo or a revision."""
    if kind == "model":
        model = load_catalog().get(catalog_id)
        if model is not None:
            return Asset("model", model.id, model.repo, model.revision, model.display_name, model.disk_gb,
                         model.license, model.gated)  # fmt: skip
    else:
        for ds in load_dataset_catalog().datasets:
            if ds.id == catalog_id and ds.security != "rejected":
                return Asset("dataset", ds.id, ds.repo, ds.revision, ds.display_name, ds.disk_gb, ds.license,
                             ds.gated)  # fmt: skip
    raise DownloadUnknownError("not a catalog asset", params={})


def planned_bytes(size_gb: float) -> int:
    """The proxy's hard budget for an asset of ``size_gb`` (security §5: size * 1.05 + 100 MB)."""
    return int(size_gb * GIB * 1.05) + 100 * 1024 * 1024


@dataclass
class Download:
    download_id: str
    asset: Asset
    run_id: str
    status: str = "queued"  # queued | downloading | verifying | installing | completed | failed
    error: dict[str, Any] | None = None
    progress: dict[str, int] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        return {"download_id": self.download_id, **self.asset.to_api(), "status": self.status,
                "progress": self.progress, "error": self.error}  # fmt: skip


class DownloadService:
    def __init__(
        self, root: Path, *, emit: Callable[[str, dict[str, Any]], None], token: Callable[[], str | None],
        launcher: Any = None,
    ) -> None:  # fmt: skip
        self.root = root
        self._emit = emit
        self._token = token  # decrypts the user's HF token from Settings only when a job starts
        self._launcher = launcher
        self._lock = threading.Lock()
        self.downloads: dict[str, Download] = {}

    # ------------------------------------------------------------------------------------------ checks
    def check_budget(self, asset: Asset) -> int:
        """Planned bytes, or NET-004 (admin cap) / DISK-001 (free disk minus the reserve) before anything runs."""
        planned = planned_bytes(asset.size_gb)
        policy = read_policy()
        cap = (policy or {}).get("download", {}).get("max_gb") if isinstance(policy, dict) else None
        if isinstance(cap, int | float) and not isinstance(cap, bool) and planned > cap * GIB:
            raise DownloadQuotaError("over the admin download cap", params={
                "size_gb": round(planned / GIB, 1), "quota_gb": round(float(cap), 1)})  # fmt: skip
        disk = read_disk(self.root)
        shared = effective_profile(os.environ.get("VLM_PROFILE"), policy) == "shared_dev"
        reserve = max(50.0, 0.10 * disk.total_gb) if shared else 20.0
        avail = disk.free_gb - reserve
        if planned > avail * GIB:
            need = {"need_gb": round(planned / GIB, 1), "avail_gb": round(max(avail, 0.0), 1), "resumable": False}
            raise DownloadDiskError("not enough free disk", params=need)
        return planned

    def installed(self, asset: Asset) -> bool:
        return snapshot_dir(self.root, asset.kind, asset.repo, asset.revision).is_dir()

    # ------------------------------------------------------------------------------------------ start
    def start(self, kind: Kind, catalog_id: str, *, confirm: bool = False, check: bool = False) -> dict[str, Any]:
        """Download ``catalog_id``, or with ``check`` verify a snapshot already there (plan only, no blob)."""
        asset = resolve(kind, catalog_id)
        if self.installed(asset):
            if verified(self.root, asset.kind, asset.repo, asset.revision):
                return {**asset.to_api(), "status": "completed", "verified": True, "download_id": None}
            if not check:  # there, but never checked by the app (a manual copy, an old cache): say so (P3)
                return {**asset.to_api(), "status": "completed", "verified": False, "download_id": None}
            return self._launch(asset, planned_bytes(0.05), plan_only=True)
        planned = self.check_budget(asset)
        if planned > CONFIRM_GB * GIB and not confirm:
            return {**asset.to_api(), "needs_confirm": True, "threshold_gb": CONFIRM_GB}
        return self._launch(asset, planned, plan_only=False)

    def _launch(self, asset: Asset, planned: int, *, plan_only: bool) -> dict[str, Any]:
        stamp = time.strftime("%Y%m%d-%H%M")
        part = _ID_PART.sub("-", asset.catalog_id.lower()).strip("-")[:30] or "asset"
        run_id = f"{stamp}-dl-{part}-{secrets.token_hex(2)}"
        self._take_active(run_id)
        download = Download(download_id=run_id, asset=asset, run_id=run_id)
        with self._lock:
            self.downloads[run_id] = download
        threading.Thread(target=self._run, args=(download, planned, plan_only), name=f"download-{run_id}",
                         daemon=True).start()  # fmt: skip
        return download.to_api()

    def _take_active(self, run_id: str) -> None:
        """One download at a time (O_EXCL); a lock whose holder is certainly gone (killed app) is replaced (P2)."""
        active = self.root / ACTIVE_REL
        active.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"download_id": run_id, "pid": os.getpid(), **process_identity(), "ts": time.time()})
        for attempt in range(2):
            try:
                fd = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                try:
                    held = read_small_json(active, max_bytes=4096)
                except (OSError, ValueError):
                    held = None
                if attempt == 0 and isinstance(held, dict) and holder_gone(held) is True:
                    active.unlink(missing_ok=True)  # a racing peer that also saw it gone loses the next O_EXCL
                    continue
                raise DownloadBusyError("another download is running", params={"minutes": 0}) from None
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            return

    # ------------------------------------------------------------------------------------------ the job
    def _set(self, d: Download, status: str, **extra: Any) -> None:
        d.status = status
        self._emit(d.download_id, {"type": "status", "status": status, **extra})

    def _run(self, d: Download, planned: int, plan_only: bool = False) -> None:
        run_dir = self.root / "outputs" / "runs" / d.run_id
        try:
            run_dir.mkdir(parents=True)
            pins = pinned_hashes(self.root, d.asset.catalog_id, d.asset.revision) if d.asset.kind == "model" else {}
            code = sorted(n for n in pins if n.endswith(".py"))
            spec = {"kind": d.asset.kind, "repo": d.asset.repo, "revision": d.asset.revision,
                    "remote_code_files": code, "max_bytes": planned, "plan_only": plan_only}  # fmt: skip
            (run_dir / SPEC_NAME).write_text(json.dumps(spec), encoding="utf-8")
            container_dir = f"{CONTAINER_ROOT}/outputs/runs/{d.run_id}"
            job = ContainerJob(
                name=d.run_id, image=DEFAULT_TRAIN_IMAGE, gpu=False, memory_gb=4, narrow=True, rw_dirs=(run_dir,),
                argv=("python", "-m", "vlm_pipeline.download.hub", "--run-dir", container_dir),
                download_budget=planned, hf_token=self._token(),
            )  # fmt: skip
            manifest = RunManifest(run_id=d.run_id, name=f"download {d.asset.name}", stage="data", task="download",
                                   dataset=f"{d.asset.repo}@{d.asset.revision}")  # fmt: skip
            self._set(d, "downloading")
            watcher = threading.Thread(target=self._watch_progress, args=(d, run_dir), daemon=True)
            watcher.start()
            final = run_job(job, manifest, run_dir, project_root=self.root, who="vlm-app", est_peak_gb=2.0,
                            take_gpu_lock=False, launcher=self._launcher)  # fmt: skip
            if final.status != RunStatus.COMPLETED:
                self._fail(d, {"code": "NET-001", "params": {"reason": "network"}})
                return
            self._set(d, "verifying")
            finish = verify_existing if plan_only else install
            finish(self.root, run_dir, kind=d.asset.kind, repo=d.asset.repo, revision=d.asset.revision,
                   catalog_id=d.asset.catalog_id, name=d.asset.name)  # fmt: skip
            self._set(d, "completed")
        except DownloadIntegrityError as exc:
            self._fail(d, {"code": exc.code, "params": exc.params})
        except VlmPipelineError as exc:
            self._fail(d, {"code": exc.code, "params": exc.params})
        except Exception:  # never a traceback to the UI
            logger.exception("download %s failed", d.download_id)
            self._fail(d, {"code": "SYS-000", "params": {}})
        finally:
            (self.root / ACTIVE_REL).unlink(missing_ok=True)

    def _fail(self, d: Download, error: dict[str, Any]) -> None:
        d.error = error
        self._set(d, "failed", error=error)

    def _watch_progress(self, d: Download, run_dir: Path) -> None:
        """The job's last progress line, re-sent while it downloads (read no-follow, bounded)."""
        path = run_dir / "stage" / "progress.jsonl"
        while d.status == "downloading":
            line = _last_line(path)
            if line is not None and line != d.progress:
                d.progress = line
                self._emit(d.download_id, {"type": "progress", **line})
            time.sleep(1.0)


def _last_line(path: Path, max_bytes: int = 4096) -> dict[str, int] | None:
    keys = ("files_done", "files_total", "bytes_done", "bytes_total")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        f.seek(max(0, size - max_bytes))
        tail = f.read(max_bytes).decode("utf-8", "replace").strip().splitlines()
    try:
        data = json.loads(tail[-1]) if tail else None
    except ValueError:
        return None
    if not isinstance(data, dict) or not all(isinstance(data.get(k), int) for k in keys):
        return None
    return {k: int(data[k]) for k in keys}


def template_assets(template: dict[str, Any]) -> list[tuple[Kind, str]]:
    """(kind, catalog id) of every catalog model / dataset a template's workflow picks."""
    wanted: list[tuple[Kind, str]] = []
    by_repo = {d.repo: d.id for d in load_dataset_catalog().datasets}
    for node in template.get("workflow", {}).get("nodes", []):
        params = node.get("params") or {}
        if node.get("type") == "model.hf_pick" and isinstance(params.get("model"), str):
            wanted.append(("model", params["model"]))
        elif node.get("type") == "data.hf_load" and params.get("repo") in by_repo:
            wanted.append(("dataset", by_repo[params["repo"]]))
    return wanted


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - install.sh, inside the app container
    """``python -m vlm_pipeline.app.downloads --template invoice_to_json``: the SAME download path as the app."""
    import argparse

    from vlm_pipeline.app.secret_store import SecretStore
    from vlm_pipeline.common.paths import project_root

    parser = argparse.ArgumentParser(prog="python -m vlm_pipeline.app.downloads")
    parser.add_argument("--template", required=True)
    parser.add_argument("--yes-large", action="store_true", help="the user confirmed downloads over the threshold")
    args = parser.parse_args(argv)
    root = project_root()
    template = json.loads((root / "configs" / "templates" / f"{Path(args.template).name}.json").read_text("utf-8"))
    store = SecretStore(root, create_key=False)
    service = DownloadService(root, emit=lambda _i, e: print(json.dumps(e), flush=True),
                              token=lambda: store.reveal("HF_TOKEN"))  # fmt: skip
    failed = 0
    for kind, catalog_id in template_assets(template):
        reply = service.start(kind, catalog_id, confirm=args.yes_large)  # never confirmed for the user (P1)
        if reply.get("needs_confirm"):
            print(f"{kind} {catalog_id}: {reply['size_gb']} GB is over {reply['threshold_gb']} GB; "
                  "download it from the app (it asks first)", flush=True)  # fmt: skip
            failed += 1
            continue
        download_id = reply.get("download_id")
        while download_id and service.downloads[download_id].status not in ("completed", "failed"):
            time.sleep(1.0)
        status = service.downloads[download_id].status if download_id else reply.get("status")
        print(f"{kind} {catalog_id}: {status}", flush=True)
        failed += status != "completed"
    return 1 if failed else 0


__all__ = ["CONFIRM_GB", "Asset", "Download", "DownloadService", "planned_bytes", "resolve", "template_assets"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
