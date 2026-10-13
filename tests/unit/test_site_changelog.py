"""/thay-doi comes from git tags only: v<semver> tags with their annotation; nothing else reaches the public page."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

from vlm_pipeline.common.paths import project_root


def _changelog() -> ModuleType:
    path = project_root() / "site" / "scripts" / "changelog.py"
    spec = importlib.util.spec_from_file_location("site_changelog", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com", "PATH": "/usr/bin:/bin"}  # fmt: skip
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)  # noqa: S603,S607


def test_only_semver_tags_with_their_annotation_newest_first(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "feat(app): internal subject naming bao-backend")
    _git(tmp_path, "tag", "-a", "v1.0.0", "-m", "First release.\n\n- Node graph editor")
    _git(tmp_path, "tag", "-a", "internal-spike", "-m", "never public")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "fix: more")
    _git(tmp_path, "tag", "v1.0.1")  # lightweight: no notes, still a release
    found = _changelog().releases(tmp_path)
    assert [r["version"] for r in found] in (["1.0.1", "1.0.0"], ["1.0.0", "1.0.1"])  # same-second dates tie
    notes = {r["version"]: r["notes"] for r in found}
    assert notes["1.0.0"] == "First release.\n\n- Node graph editor"
    assert "bao-backend" not in str(found)  # commit subjects are never published


def test_no_git_or_no_tag_means_no_release(tmp_path: Path) -> None:
    assert _changelog().releases(tmp_path / "missing") == []
    _git(tmp_path, "init", "-q")
    assert _changelog().releases(tmp_path) == []
