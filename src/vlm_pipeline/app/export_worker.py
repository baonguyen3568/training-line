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
import logging
import os
import re
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

logger = logging.getLogger(__name__)
COPY_SUFFIXES = frozenset({".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken"})
NEVER_COPY = frozenset({"run.json", "merge_report.json", "dataset.json", "problems.json", "export.json"})


class ExportLicenseUnknownError(VlmPipelineError):
    """A user-added model whose card names no licence: usable in the app, never exported (ADR-021 A1, SEC-006)."""

    code: ClassVar[str] = "SEC-006"


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


class ExportIncompleteError(VlmPipelineError):
    """The export folder lacks the files that make it a model (config.json, weights): never reported completed."""

    code: ClassVar[str] = "EXP-004"


_SHA256_NAME = re.compile(r"[0-9a-f]{64}")


def _copy_verified(source: Path, dest: Path, *, expect: str | None) -> bool:
    """Copy ``source``'s content; with ``expect`` (a sha256) the bytes must hash to it, else nothing is kept.

    A damaged cache blob is left out: the completeness check then names the missing file (EXP-004).
    """
    digest = hashlib.sha256()
    with source.open("rb") as fin, dest.open("xb") as fout:
        for chunk in iter(lambda: fin.read(8 << 20), b""):
            digest.update(chunk)
            fout.write(chunk)
    if expect is not None and digest.hexdigest() != expect:
        dest.unlink()
        logger.warning("export: blob %s does not match its sha256 name; left out", source.name[:12])
        return False
    return True


def _copy_allowed(src: Path, dst: Path, *, blobs: tuple[Path, ...] = ()) -> list[str]:
    """Copy the model files of ``src`` (plain files) into ``dst``; returns the names copied.

    An HF hub snapshot is made of links into its repo's ``blobs/<etag>``, which may link again into the shared,
    deduplicated store ``.cache/hf/hub/blobs/<xx>/<key>`` (tester, release 1.0). A link whose final target is a plain
    file inside one of ``blobs`` is copied as its content; when the repo blob's name is a sha256 (HF LFS etag) the
    bytes must hash to it, and a shared-store file is only taken with such a hash to check. Other links are skipped.
    """
    copied = []
    for path in sorted(src.iterdir()):
        if path.suffix not in COPY_SUFFIXES or path.name in NEVER_COPY:
            continue
        source, expect = path, None
        if path.is_symlink():
            target = Path(os.path.realpath(path))
            if not any(target.is_relative_to(b) for b in blobs):
                continue
            source = target
            # The first hop is the repo blob, named by HF's sha256 of the content for LFS files (tester): that is the
            # hash to check. The shared store's own names are dedupe keys, not content hashes.
            first = path.parent / path.readlink()
            expect = first.name if _SHA256_NAME.fullmatch(first.name) else None
            if expect is None and blobs[-1:] and target.is_relative_to(blobs[-1]):
                continue  # into the shared store with no published hash to check it against: left out
        if not stat.S_ISREG(source.lstat().st_mode):
            continue
        if _copy_verified(source, dst / path.name, expect=expect):
            copied.append(path.name)
    return copied


def _hub_blobs(root: Path, src: Path) -> tuple[Path, ...]:
    """Where the links of hub snapshot ``src`` may lead: its repo's ``blobs/``, then the shared store (resolved).

    Empty when ``src`` is not in the hub. The shared ``.cache/hf/hub/blobs`` store is last. huggingface_hub writes it
    (marker ``.huggingface-shared-blobs``, ``<name>.refs``; hf_xet downloads): its names are internal dedupe keys that
    may change between versions, never a checksum, so a file there is only taken when the repo blob linking to it
    names the content's sha256 (the LFS etag).
    """
    base = root.resolve()
    real = src.resolve()
    if not real.is_relative_to(base):
        return ()
    rel = real.relative_to(base)
    if rel.parts[:3] != (".cache", "hf", "hub") or len(rel.parts) < 4 or rel.parts[3] == "blobs":
        return ()
    hub = base.joinpath(*rel.parts[:3])
    return (Path(os.path.realpath(hub / rel.parts[3] / "blobs")), Path(os.path.realpath(hub / "blobs")))


def _check_complete(folder: Path, job: NodeJob) -> None:
    """A model folder has config.json and at least one weights file (tester, release 1.0 smoke)."""
    missing = [name for name, ok in (
        ("config.json", (folder / "config.json").is_file()),
        ("*.safetensors", any(folder.glob("*.safetensors"))),
    ) if not ok]  # fmt: skip
    index = folder / "model.safetensors.index.json"
    if index.is_file():  # sharded weights: every shard the index names must be there, or loading fails
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        except (ValueError, AttributeError):
            shards = {"model.safetensors.index.json"}
        missing += sorted(str(n) for n in shards if not (isinstance(n, str) and (folder / n).is_file()))
    if missing:
        raise ExportIncompleteError(
            "export folder has no model", params={"node": job.node_id, "files": ", ".join(missing)}
        )


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
    src = worker.model_dir(model)
    unsupported = {"node": job.node_id, "model": entry.display_name}
    if entry.tier == "user" and entry.license == "unknown":  # ADR-021 A1: trains, never leaves the app
        raise ExportLicenseUnknownError("no licence named for this model", params={
            "node": job.node_id, "name": entry.display_name, "repo": entry.repo})  # fmt: skip
    if entry.remote_code != "none" or portable.remote_code_findings(src):
        raise ExportUnsupportedError("model runs its own code", params=unsupported)
    dest = _destination(worker, job, str(getattr(job.params, "dest")))  # noqa: B009
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{secrets.token_hex(4)}")
    tmp.mkdir()
    try:
        upstream = worker._catalog_model_path(entry)
        _copy_allowed(src, tmp, blobs=_hub_blobs(worker.root, src))
        _check_complete(tmp, job)
        portable.neutralize_name_or_path(tmp, f"{entry.repo}@{entry.revision}")
        portable.follow_base_use_cache(tmp, upstream)  # an export is for inference: never a training-time false
        if portable.remote_code_findings(tmp) or portable.absolute_path_findings(tmp, upstream=upstream):
            raise ExportUnsupportedError("export is not portable", params=unsupported)
        base_ref = model.hash if "@" in model.hash.split(":", 1)[0] else None
        components, mods, gaps = notices.lineage(
            src, root=worker.root, catalog=worker.catalog, datasets=load_dataset_catalog(), base_ref=base_ref
        )
        components = notices.fill_notices(components, tool=i18n.t("app.name", "en"), date=time.strftime("%Y-%m-%d"))
        try:
            attribution = notices.write_notices(
                tmp, root=worker.root, components=components, modifications=mods, gaps=gaps, catalog=worker.catalog,
                upstream_dir=upstream,
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
