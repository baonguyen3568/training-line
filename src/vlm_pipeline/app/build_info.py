"""What code this app process runs (/api/health) and the public bug-report card (security, release 1.0 checklist F).

The commit is read ONCE at startup: after a merge, /api/health still names the code that is running, so the
checklist can compare it with HEAD. The bug report carries only what security approved for a PUBLIC issue: the
error code, its catalog title, the app version, the GPU model and a short hashed reference of the run; never logs,
paths, host names or the user's run / step names.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import importlib.metadata
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_pipeline import i18n
from vlm_pipeline.common.error_catalog import load_catalog
from vlm_pipeline.common.jobrun import git_state

BUILD_COMMIT_ENV = "VLM_BUILD_COMMIT"  # set by the image build (no .git inside the app image)


@dataclass(frozen=True)
class BuildInfo:
    version: str
    commit: str | None  # full sha, or None when unknown
    dirty: bool | None  # src/configs/docker/scripts differ from the commit (None: unknown)

    @property
    def app_version(self) -> str:
        """``0.1.0+<sha12>[-dirty]`` (just the version when the commit is unknown)."""
        if self.commit is None:
            return self.version
        return f"{self.version}+{self.commit[:12]}{'-dirty' if self.dirty else ''}"

    def to_api(self) -> dict[str, Any]:
        return {"version": self.version, "commit": self.commit, "dirty": self.dirty, "app_version": self.app_version}


def read_build_info(root: Path) -> BuildInfo:
    try:
        version = importlib.metadata.version("vlm-pipeline")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0"
    commit, dirty = git_state(root)
    if commit is None and (baked := os.environ.get(BUILD_COMMIT_ENV, "").strip()):
        commit, dirty = baked, False  # an image is built from a commit; its files cannot change
    return BuildInfo(version=version, commit=commit, dirty=dirty)


@functools.cache
def gpu_name() -> str | None:
    """The first GPU's model name (no serial, no driver detail), or None without a GPU."""
    from vlm_pipeline.common.gpu_mem import query_gpus

    gpus = query_gpus()
    return gpus[0].name if gpus else None


REF_KEY_REL = Path("outputs") / "system" / "ref_code.key"  # one purpose, never the D-7 secrets.key (security H1)
_REF_KEY_BYTES = 32


def ref_key(root: Path) -> bytes:
    """This install's random key for ref codes, created once (0600, written whole then linked: no partial reads)."""
    path = root / REF_KEY_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(_REF_KEY_BYTES))
        try:
            os.link(tmp, path)  # first writer wins; a concurrent one reads the winner's key below
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as f:
        key = f.read(_REF_KEY_BYTES + 1)
    if len(key) != _REF_KEY_BYTES:
        raise ValueError("ref_code.key is damaged")
    return key


def ref_code(run_id: str, key: bytes) -> str:
    """8 hex of HMAC-SHA256(install key, run_id): lets the team ask about "that run".

    A dictionary of likely run names (CLI run ids carry the user's own tag) cannot reverse it without the key, which
    never leaves the machine (security H1).
    """
    return hmac.new(key, run_id.encode(), hashlib.sha256).hexdigest()[:8]


def bug_report(
    code: str, locale: str, info: BuildInfo, *, run_id: str | None = None, key: bytes | None = None
) -> dict[str, str]:
    """The public bug-report fields; ``code`` must be a catalog code (KeyError otherwise). Absent ones are left out."""
    if code not in load_catalog():
        raise KeyError(code)
    report: dict[str, str] = {"code": code, "app_version": info.app_version}
    title = i18n.load(locale).get(f"error.{code}.title")
    if title and not i18n.placeholders(title):  # the catalog's own words, never an exception message
        report["error_title"] = title
    if (gpu := gpu_name()) is not None:
        report["gpu"] = gpu
    if run_id is not None and key is not None:
        report["ref_code"] = ref_code(run_id, key)
    return report


__all__ = ["BUILD_COMMIT_ENV", "BuildInfo", "bug_report", "gpu_name", "read_build_info", "ref_code", "ref_key"]
