"""research 2026-10-13: with $VLM_CODE_ROOT the host reads configs/ from the same tree the jobs run."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.paths import CODE_ROOT_ENV, configs_root

PYPROJECT = '[project]\nname = "vlm-pipeline"\n'


def _tree(path: Path, *, worktree: bool = False) -> Path:
    (path / "configs").mkdir(parents=True)
    (path / "pyproject.toml").write_text(PYPROJECT)
    if worktree:
        (path / ".git").write_text("gitdir: /x\n")
    return path


def test_without_a_code_root_configs_are_the_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CODE_ROOT_ENV, raising=False)
    root = _tree(tmp_path / "repo")
    assert configs_root(root) == root.resolve()


@pytest.mark.parametrize("where", [".cache/wt-research", ".cache/check-clean.123", ".worktrees/e2e"])
def test_a_team_worktree_supplies_the_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    root = _tree(tmp_path / "repo")
    code = _tree(root / where, worktree=True)
    monkeypatch.setenv(CODE_ROOT_ENV, str(code))
    assert configs_root(root) == code.resolve()


@pytest.mark.parametrize("where", ["outputs/evil", "data/wt-x", ".cache/other", "elsewhere"])
def test_anything_else_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    root = _tree(tmp_path / "repo")
    code = _tree((root / where) if where != "elsewhere" else tmp_path / "elsewhere", worktree=True)
    monkeypatch.setenv(CODE_ROOT_ENV, str(code))
    with pytest.raises(ValueError, match=CODE_ROOT_ENV):  # a job-writable folder never feeds the host's catalogs
        configs_root(root)


def test_a_worktree_without_a_git_link_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path / "repo")
    code = _tree(root / ".cache" / "wt-plain")  # no .git file
    monkeypatch.setenv(CODE_ROOT_ENV, str(code))
    with pytest.raises(ValueError, match=CODE_ROOT_ENV):  # set but not valid: never silently the root's configs
        configs_root(root)


def test_a_code_root_equal_to_the_project_is_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path / "repo")
    monkeypatch.setenv(CODE_ROOT_ENV, str(root))
    assert configs_root(root) == root.resolve()
