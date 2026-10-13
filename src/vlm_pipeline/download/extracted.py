"""Eval data a download turns into plain files (G-5, tester 2026-10-13): e.g. OCRBench parquet -> images + questions.

The catalog snapshot is only the source. A job (no network, non-root, the snapshot read-only, its run dir the only
writable place) extracts it; the HOST then checks what the job wrote before anything reaches ``data/eval/``:

1. the source file matches the catalog's ``files_sha256`` (checked on the host before the job starts, and again by
   the job itself);
2. the job's output tree holds regular files only, each name allowed (``images/<n>.jpg|png|webp``, ``questions.jsonl``,
   ``MANIFEST.sha256``), ``MANIFEST.sha256`` lists exactly those files with matching sha256, and the manifest itself
   hashes to the catalog's ``extracted_manifest_sha256`` (the extraction is deterministic: the tree is THE known one,
   not merely self-consistent; security 2026-10-13);
3. the checked tree is renamed into ``<root>/data/eval/<id>-<rev8>/`` through a temporary sibling (all or none).

An installed copy that no longer checks out (a changed image, an interrupted copy) is moved aside to
``.<name>.invalid-<time>`` (kept for a look, never deleted) and extracted again: "Download" always repairs it.

``data/eval/`` is held-out data: DATA-007 keeps it out of training and calibration.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.common.catalog import DatasetEntry, load_dataset_catalog
from vlm_pipeline.common.local_cache import eval_data_dir as _eval_data_dir
from vlm_pipeline.download.install import DownloadIntegrityError, snapshot_dir

MANIFEST = "MANIFEST.sha256"
_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
_NAME = re.compile(r"(images/\d{1,6}\.(?:jpg|png|webp)|questions\.jsonl)")  # the vLLM job: PNG/JPEG/WEBP
_LINE = re.compile(r"([0-9a-f]{64})  (\S+)")


@dataclass(frozen=True)
class ExtractSpec:
    """How one catalog dataset becomes eval files: its source file in the snapshot and the job's module."""

    catalog_id: str
    source: str  # path inside the snapshot, listed in the catalog's files_sha256
    module: str  # ``python -m <module> --parquet <source> --out <dir>`` inside the job

    def entry(self) -> DatasetEntry:
        entry = next(d for d in load_dataset_catalog().datasets if d.id == self.catalog_id)
        if self.source not in entry.files_sha256 or entry.extracted_manifest_sha256 is None:
            raise ValueError(f"{self.catalog_id}: no catalog sha256 for {self.source} or its extracted manifest")
        return entry


EXTRACTED: dict[str, ExtractSpec] = {
    "ocrbench": ExtractSpec("ocrbench", "data/test-00000-of-00001.parquet", "vlm_pipeline.eval.ocrbench_extract"),
}


def eval_data_dir(root: Path, catalog_id: str) -> Path:
    """``<root>/data/eval/<id>-<rev8>``: where the extracted files live (what eval nodes read)."""
    return _eval_data_dir(root, EXTRACTED[catalog_id].entry())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source(root: Path, catalog_id: str) -> Path:
    """The snapshot's source file, a regular file whose sha256 is the catalog's; DownloadIntegrityError otherwise."""
    spec = EXTRACTED[catalog_id]
    entry = spec.entry()
    path = snapshot_dir(root, "dataset", entry.repo, entry.revision) / spec.source
    params: dict[str, object] = {"name": entry.display_name, "reason": "mismatch"}
    try:
        regular = stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        regular = False
    if not regular or _sha256(path) != entry.files_sha256[spec.source]:
        raise DownloadIntegrityError(f"{spec.source}: not the catalog's file", params=params)
    return path


def _manifest(tree: Path) -> dict[str, str]:
    """``name -> sha256`` from MANIFEST.sha256 (bounded, no link, sha256sum format, every name allowed, no dup)."""
    path = tree / MANIFEST
    if not stat.S_ISREG(os.lstat(path).st_mode) or path.stat().st_size > _MANIFEST_MAX_BYTES:
        raise ValueError("manifest is not a small regular file")
    listed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LINE.fullmatch(line)
        if match is None or not _NAME.fullmatch(match[2]) or match[2] in listed:
            raise ValueError("malformed manifest line")
        listed[match[2]] = match[1]
    return listed


def _tree_files(tree: Path) -> dict[str, Path]:
    """Every file under ``tree`` (no link followed); anything but a plain file or directory is refused."""
    found: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            mode = os.lstat(path).st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"not a plain file: {name}")
            found[path.relative_to(tree).as_posix()] = path
    return found


def verify_tree(tree: Path, *, name: str, manifest_sha256: str) -> None:
    """The tree is exactly what its manifest lists, hash for hash, and the manifest is the pinned one; else NET-003."""
    params: dict[str, object] = {"name": name, "reason": "mismatch"}
    try:
        if _sha256(tree / MANIFEST) != manifest_sha256:
            raise ValueError("not the expected extraction (manifest hash)")
        listed = _manifest(tree)
        files = _tree_files(tree)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise DownloadIntegrityError(f"extracted data: {exc}", params=params) from None
    files.pop(MANIFEST, None)
    if not listed or set(files) != set(listed) or "questions.jsonl" not in listed:
        raise DownloadIntegrityError("extracted files differ from the manifest", params=params)
    for rel, path in files.items():
        if _sha256(path) != listed[rel]:
            raise DownloadIntegrityError(f"hash mismatch: {rel}", params=params)


def extracted_ok(root: Path, catalog_id: str) -> bool:
    """The installed eval files check out in full (same check as at install; ~70 MB of files for OCRBench)."""
    entry = EXTRACTED[catalog_id].entry()
    target = eval_data_dir(root, catalog_id)
    if target.is_symlink() or not target.is_dir():
        return False
    try:
        verify_tree(target, name=entry.display_name, manifest_sha256=str(entry.extracted_manifest_sha256))
    except DownloadIntegrityError:
        return False
    return True


def _move_aside(target: Path) -> Path:
    """A copy that no longer checks out: renamed out of the way (kept, never deleted) so it can be extracted again."""
    aside = target.with_name(f".{target.name}.invalid-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}")
    target.rename(aside)
    return aside


def install_extracted(root: Path, catalog_id: str, tree: Path) -> Path:
    """Verify the job's ``tree`` and rename it into ``data/eval/``; an existing valid copy is kept as is."""
    entry = EXTRACTED[catalog_id].entry()
    verify_tree(tree, name=entry.display_name, manifest_sha256=str(entry.extracted_manifest_sha256))
    target = eval_data_dir(root, catalog_id)
    if target.is_symlink() or target.exists():
        if extracted_ok(root, catalog_id):  # a good copy is kept as is
            shutil.rmtree(tree, ignore_errors=True)
            return target
        _move_aside(target)  # damaged or partial: out of the way, kept
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    try:
        shutil.move(str(tree), tmp)  # a rename on the data root
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


__all__ = [
    "EXTRACTED",
    "MANIFEST",
    "ExtractSpec",
    "check_source",
    "eval_data_dir",
    "extracted_ok",
    "install_extracted",
    "verify_tree",
]
