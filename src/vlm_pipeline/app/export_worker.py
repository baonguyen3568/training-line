"""Node ``export.hf_dir`` (M1b B-2, ADR-012): a shareable model folder with its licences, notices and model card.

Host job, CPU only. Native models only for now (ADR-012 B3): a model that runs its own code is refused, judged on the
folder's REAL content (any ``auto_map`` / ``*.py``), whatever the catalog says. Steps, all into a temporary folder
next to the destination and renamed at the end (never a half-written export):
1. copy the weights + configs + tokenizer/processor files (allowlist of suffixes, regular files only);
2. ``_name_or_path`` -> ``repo@revision`` (never a machine path), then re-check: no remote code, no machine path;
3. licence texts, NOTICE, README.md, ATTRIBUTION.json (``export.notices``; no approved licence text -> EXP-001);
4. no credential-shaped string anywhere in the folder (``common.secret_scan``) -> else EXP-003, nothing written.
The node output is a small receipt (``export.json``) in its run dir; the exported folder belongs to the user.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from vlm_pipeline import i18n
from vlm_pipeline.common.catalog import load_dataset_catalog
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.secret_scan import scan_files, walk
from vlm_pipeline.export import notices, portable
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker

COPY_SUFFIXES = frozenset({".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken"})
NEVER_COPY = frozenset({"run.json", "merge_report.json", "dataset.json", "problems.json", "export.json"})


class ExportUnsupportedError(VlmPipelineError):
    """The model runs its own code: not exportable as a portable folder yet (ADR-012 B3)."""

    code: ClassVar[str] = "EXP-002"


class ExportLicenseMissingError(VlmPipelineError):
    code: ClassVar[str] = "EXP-001"


class ExportSecretError(VlmPipelineError):
    """Something in the export looks like a credential (bao-security token shapes): it is never written out."""

    code: ClassVar[str] = "EXP-003"


class ExportDestinationError(VlmPipelineError):
    """The destination must be a folder under outputs/ (never inside the code, the cache or another run)."""

    code: ClassVar[str] = "CFG-005"


def _copy_allowed(src: Path, dst: Path) -> list[str]:
    copied = []
    for path in sorted(src.iterdir()):
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or path.suffix not in COPY_SUFFIXES or path.name in NEVER_COPY:
            continue
        shutil.copy2(path, dst / path.name)
        copied.append(path.name)
    return copied


def _destination(worker: RealWorker, job: NodeJob, raw: str) -> Path:
    outputs = (worker.root / "outputs").resolve()
    dest = (worker.root / raw).resolve()
    forbidden = (outputs / "graph", outputs / "runs", outputs / "system")
    if not dest.is_relative_to(outputs) or dest == outputs or any(dest.is_relative_to(f) for f in forbidden):
        raise ExportDestinationError("export destination must be a new folder under outputs/", params={
            "node": job.node_id, "param": "dest", "value": raw, "rule": "outputs_folder"})  # fmt: skip
    if dest.exists() and any(dest.iterdir()):  # never mix two exports: a new sibling folder instead
        dest = dest.with_name(f"{dest.name}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}")
    return dest


def run_export(worker: RealWorker, job: NodeJob) -> NodeResult:
    model = job.inputs["model"]
    entry = worker.base_of(model)
    src = worker.host_path(model.path)
    unsupported = {"node": job.node_id, "model": entry.display_name}
    if entry.remote_code != "none" or portable.remote_code_findings(src):
        raise ExportUnsupportedError("model runs its own code", params=unsupported)
    dest = _destination(worker, job, str(getattr(job.params, "dest")))  # noqa: B009
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{secrets.token_hex(4)}")
    tmp.mkdir()
    try:
        _copy_allowed(src, tmp)
        portable.neutralize_name_or_path(tmp, f"{entry.repo}@{entry.revision}")
        if portable.remote_code_findings(tmp) or portable.absolute_path_findings(tmp):
            raise ExportUnsupportedError("export is not portable", params=unsupported)
        base_ref = model.hash if "@" in model.hash.split(":", 1)[0] else None
        components, mods, gaps = notices.lineage(
            src, root=worker.root, catalog=worker.catalog, datasets=load_dataset_catalog(), base_ref=base_ref
        )
        components = notices.fill_notices(components, tool=i18n.t("app.name", "en"), date=time.strftime("%Y-%m-%d"))
        try:
            attribution = notices.write_notices(
                tmp, root=worker.root, components=components, modifications=mods, gaps=gaps, catalog=worker.catalog,
                upstream_dir=worker._catalog_model_path(entry),
            )  # fmt: skip
        except notices.LicenseTextMissingError as exc:
            raise ExportLicenseMissingError(str(exc), params={"license_ids": ", ".join(exc.licenses)}) from exc
        if leaks := scan_files(walk([tmp]), base=tmp):  # the user shares this folder: never with a token inside
            files, kinds = sorted({leak.path for leak in leaks}), sorted({leak.kind for leak in leaks})
            raise ExportSecretError(  # names and pattern kinds only: never the value or text around it
                f"credential-shaped string ({', '.join(kinds)}) in the export",
                params={"node": job.node_id, "files": ", ".join(files)},
            )
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # our own temporary folder only
        raise
    job.run_dir.mkdir(parents=True, exist_ok=True)
    receipt = job.run_dir / "export.json"
    body = {"dest": dest.relative_to(worker.root).as_posix(), "files": attribution.files, "gaps": list(gaps)}
    receipt.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256((dest / "ATTRIBUTION.json").read_bytes()).hexdigest()
    return NodeResult(outputs={"artifact": ArtifactRef(type="ExportArtifact", path=str(receipt), hash=digest)})


__all__ = [
    "ExportDestinationError",
    "ExportLicenseMissingError",
    "ExportSecretError",
    "ExportUnsupportedError",
    "run_export",
]
