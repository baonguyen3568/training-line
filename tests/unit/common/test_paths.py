"""Unit tests for vlm_pipeline.common.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.paths import PROJECT_ROOT_ENV, project_root


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    return tmp_path


def test_finds_repo_by_walking_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    repo = _repo(tmp_path)
    assert project_root(repo / "src" / "pkg" / "x.py") == repo.resolve()


def test_ignores_unrelated_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    repo = _repo(tmp_path)
    other = repo / "src" / "pkg"
    (other / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    assert project_root(other / "x.py") == repo.resolve()


def test_env_override_must_be_this_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path / "repo")
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(repo))
    assert project_root() == repo.resolve()
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))  # wrong directory, e.g. $HOME
    with pytest.raises(FileNotFoundError, match="not the vlm-pipeline repository"):
        project_root()


def test_real_repo_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    assert (project_root() / "src" / "vlm_pipeline").is_dir()
