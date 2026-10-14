"""Host side of a catalog download (P018, security download-egress §4): verify the job's staging dir, then install.

The job is untrusted: everything it wrote is re-checked here, in the app process, before one byte reaches the shared
``.cache/hf/hub``. Steps, all-or-nothing:

1. walk ``<run>/stage/files`` without following links: regular files only, each name allowed by the shared rules
   (``common.download_rules``), the set equal to the job's plan;
2. re-hash every file: an LFS file must match the hub's sha256 from the plan (integrity); a remote-code ``.py`` must
   match the security pin (``configs/models/pinned/<id>@<rev8>.sha256``, authenticity); any mismatch refuses all;
3. move the files into ``.cache/hf/hub/<models|datasets>--org--name/snapshots/<revision>/`` (plain files) through a
   temporary sibling renamed into place, so a reader sees the whole snapshot or none of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any, ClassVar

from vlm_pipeline.common.download_rules import Kind, check_repo, name_ok, wanted
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.paths import configs_root
from vlm_pipeline.common.runinfo import read_small_json


class DownloadIntegrityError(VlmPipelineError):
    """What the job left does not match the plan, the hub's hashes or the security pin: nothing is installed."""

    code: ClassVar[str] = "NET-003"


def pinned_hashes(root: Path, catalog_id: str, revision: str) -> dict[str, str]:
    """``name -> sha256`` from ``configs/models/pinned/<id>@<rev8>.sha256`` (``sha256sum`` format); {} if none."""
    path = configs_root(root) / "configs" / "models" / "pinned" / f"{catalog_id}@{revision[:8]}.sha256"
    if not path.is_file():
        return {}
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        if digest and name:
            pins[name.strip()] = digest.strip()
    return pins


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """Git's object id of a file (sha1 of "blob <size>", a NUL, then the content): the hub's ``blob_id`` (non-LFS)."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())  # noqa: S324 - git's id, compared not trusted
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(files: Path) -> dict[str, Path]:
    """Map each name under ``files`` to its path; anything but a plain file or directory refuses all."""
    found: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(files, followlinks=False):
        for entry in [*dirnames, *filenames]:
            path = Path(dirpath) / entry
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise DownloadIntegrityError(f"not a plain file: {entry}")
            found[path.relative_to(files).as_posix()] = path
    return found


PLAN_MAX_BYTES = 16 * 1024 * 1024  # a dataset with many files has a long plan; still bounded (security D2)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def _read_plan(stage: Path, params: dict[str, object]) -> dict[str, dict[str, Any]]:
    """The job's plan, read no-follow / regular / bounded and schema-checked: ``name -> entry`` (security D2)."""
    try:
        raw = read_small_json(stage / "plan.json", max_bytes=PLAN_MAX_BYTES)
        files = raw["files"] if isinstance(raw, dict) else None
        if not isinstance(files, list):
            raise TypeError
    except (OSError, ValueError, KeyError, TypeError):
        raise DownloadIntegrityError("no readable plan", params=params) from None
    planned: dict[str, dict[str, Any]] = {}
    for f in files:
        ok = (
            isinstance(f, dict)
            and isinstance(f.get("name"), str) and name_ok(f["name"])
            and isinstance(f.get("size"), int) and not isinstance(f.get("size"), bool) and f["size"] >= 0
            and (f.get("sha256") is None or (isinstance(f["sha256"], str) and _SHA256.fullmatch(f["sha256"])))
            and (f.get("git_sha1") is None or (isinstance(f["git_sha1"], str) and _SHA1.fullmatch(f["git_sha1"])))
        )  # fmt: skip
        if not ok or f["name"] in planned:
            raise DownloadIntegrityError("malformed plan", params=params)
        planned[f["name"]] = f
    return planned


def _check_tree(files_dir: Path, planned: dict[str, dict[str, Any]], *, kind: Kind, pins: dict[str, str],
                params: dict[str, object], archives: bool = False) -> dict[str, Path]:  # fmt: skip
    """Every file of ``files_dir`` is planned, allowed and hashes to what the plan (or the pin) says."""
    files = _walk(files_dir)
    if set(files) != set(planned):
        raise DownloadIntegrityError("files differ from the plan", params=params)
    remote_code = list(pins)
    for rel, path in files.items():
        if not wanted(rel, kind, remote_code, archives=archives):
            raise DownloadIntegrityError(f"file not allowed: {rel}", params=params)
        entry = planned[rel]
        if rel.endswith(".py"):  # remote code: the security pin, never the hub's word
            if pins.get(rel) is None or _sha256(path) != pins[rel]:
                raise DownloadIntegrityError(f"unpinned or changed code: {rel}", params=params)
        elif isinstance(entry.get("sha256"), str):  # LFS file: the hub's sha256 of the content
            if _sha256(path) != entry["sha256"]:
                raise DownloadIntegrityError(f"hash mismatch: {rel}", params=params)
        elif isinstance(entry.get("git_sha1"), str):  # small file in git: its blob id at the pinned commit
            if _git_blob_sha1(path) != entry["git_sha1"]:
                raise DownloadIntegrityError(f"hash mismatch: {rel}", params=params)
        else:
            raise DownloadIntegrityError(f"no hash to check: {rel}", params=params)
    return files


def verify(stage: Path, *, kind: Kind, pins: dict[str, str], name: str, archives: bool = False) -> dict[str, Path]:
    """The verified ``name -> path`` of the staged files, or DownloadIntegrityError (params: the asset's name).

    ``archives``: a user dataset's WebDataset shards are allowed (read later by a network-none job, H5b).
    """
    params: dict[str, object] = {"name": name, "reason": "mismatch"}
    planned = _read_plan(stage, params)
    return _check_tree(stage / "files", planned, kind=kind, pins=pins, params=params, archives=archives)


INSTALLED_REL = Path("outputs") / "system" / "downloads" / "installed"


def record_path(root: Path, kind: Kind, repo: str, revision: str) -> Path:
    """The app's record of a verified snapshot (security P3): written by the host, never by a job."""
    return root / INSTALLED_REL / f"{kind}--{repo.replace('/', '--')}@{revision}.json"


def _write_record(root: Path, kind: Kind, repo: str, revision: str, planned: dict[str, dict[str, Any]]) -> None:
    path = record_path(root, kind, repo, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = [{k: f.get(k) for k in ("name", "size", "sha256", "git_sha1")} for f in planned.values()]
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps({"files": sorted(files, key=lambda f: str(f["name"]))}, indent=1), encoding="utf-8")
    tmp.chmod(0o644)
    tmp.replace(path)


def verified(root: Path, kind: Kind, repo: str, revision: str) -> bool:
    """A cheap re-check of a snapshot the app installed: its record exists and names + sizes still match.

    "verified" means: matches the install record by name and size. The full hash check is the Check action
    (``verify_existing``); the cache is written by the host only.
    """
    target = snapshot_dir(root, kind, repo, revision)
    try:
        record = read_small_json(record_path(root, kind, repo, revision), max_bytes=PLAN_MAX_BYTES)
        files = {str(f["name"]): int(f["size"]) for f in record["files"]}  # type: ignore[index]
        on_disk = {rel: path.stat().st_size for rel, path in _walk(target).items()}
    except (OSError, ValueError, KeyError, TypeError, DownloadIntegrityError):
        return False
    return files == on_disk


def verify_existing(root: Path, run_dir: Path, *, kind: Kind, repo: str, revision: str, catalog_id: str,
                    name: str) -> Path:  # fmt: skip
    """The Check action (plan-only job): an existing snapshot must match the hub's plan, then it gets its record."""
    target = snapshot_dir(root, kind, repo, revision)
    params: dict[str, object] = {"name": name, "reason": "cached_differs"}
    if target.is_symlink() or not target.is_dir():
        raise DownloadIntegrityError("no snapshot to check", params=params)
    planned = _read_plan(run_dir / "stage", params)
    pins = pinned_hashes(root, catalog_id, revision) if kind == "model" else {}
    _check_tree(target, planned, kind=kind, pins=pins, params=params)
    _write_record(root, kind, repo, revision, planned)
    return target


def snapshot_dir(root: Path, kind: Kind, repo: str, revision: str) -> Path:
    check_repo(repo, revision)
    prefix = "models" if kind == "model" else "datasets"
    return root / ".cache" / "hf" / "hub" / f"{prefix}--{repo.replace('/', '--')}" / "snapshots" / revision


def install(root: Path, run_dir: Path, *, kind: Kind, repo: str, revision: str, catalog_id: str, name: str) -> Path:
    """Verify ``run_dir``'s staging dir and install it as the repo's snapshot; returns the snapshot dir."""
    target = snapshot_dir(root, kind, repo, revision)
    pins = pinned_hashes(root, catalog_id, revision) if kind == "model" else {}
    params: dict[str, object] = {"name": name, "reason": "mismatch"}
    stage = run_dir / "stage"
    planned = _read_plan(stage, params)
    files = _check_tree(stage / "files", planned, kind=kind, pins=pins, params=params)
    if target.is_dir() or target.is_symlink():  # security D3: an existing snapshot is only used if it checks out
        cached = {**params, "reason": "cached_differs"}  # the user removes it (Storage panel), the app never does
        if target.is_symlink():
            raise DownloadIntegrityError("the cached snapshot is a link", params=cached)
        _check_tree(target, planned, kind=kind, pins=pins, params=cached)  # differs: never used, never overwritten
        _write_record(root, kind, repo, revision, planned)
        shutil.rmtree(stage / "files", ignore_errors=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{revision}.{secrets.token_hex(4)}.tmp")
    tmp.mkdir()
    try:
        for rel, path in files.items():
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, dest)  # a rename on the data root; a copy if .cache is another filesystem
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_record(root, kind, repo, revision, planned)
    return target


__all__ = [
    "DownloadIntegrityError",
    "install",
    "pinned_hashes",
    "record_path",
    "snapshot_dir",
    "verified",
    "verify",
    "verify_existing",
]
