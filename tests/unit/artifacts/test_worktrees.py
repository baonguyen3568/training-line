"""Unit tests for ``vlm_pipeline.artifacts.worktrees`` with a real throw-away git repo in tmp_path."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.worktrees import apply, git_runner, plan
from vlm_pipeline.common.diskguard import LockStatus

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def no_lock(_path: Path) -> LockStatus | None:
    return None


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vlm-pipeline"\n')
    _git(tmp_path, "add", "pyproject.toml")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _gate_run(root: Path, run_id: str, status: str | None) -> Path:
    _git(root, "worktree", "add", "-q", "--detach", f".worktrees/{run_id}", "HEAD")
    run_dir = root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True)
    if status is not None:
        (run_dir / "run.json").write_text(json.dumps({"run_id": run_id, "status": status}))
    return root / ".worktrees" / run_id


def _by_run(root: Path) -> dict[str, tuple[str, str | None]]:
    return {i.run_id: (i.action, i.blocked_by) for i in plan(root, lock_reader=no_lock)}


def test_plan_decisions(repo: Path) -> None:
    for run_id, status in [
        ("done", "completed"),
        ("bad", "failed"),
        ("live", "running"),
        ("paused", "interrupted"),
        ("nojson", None),
    ]:
        _gate_run(repo, run_id, status)
    _git(repo, "worktree", "add", "-q", "--detach", ".worktrees/orphan", "HEAD")  # no outputs/runs/orphan
    assert _by_run(repo) == {
        "done": ("remove", None),
        "bad": ("remove", None),
        "live": ("keep", "RUN_NOT_FINISHED"),
        "paused": ("keep", "RUN_NOT_FINISHED"),
        "nojson": ("keep", "UNKNOWN_STATUS"),
        "orphan": ("keep", "UNKNOWN_RUN"),
    }


def test_main_checkout_and_outside_worktrees_are_never_candidates(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("elsewhere") / "wt"
    _git(repo, "worktree", "add", "-q", "--detach", str(outside), "HEAD")
    assert plan(repo, lock_reader=no_lock) == []


def test_running_worktree_is_blocked_and_kept_on_apply(repo: Path) -> None:
    wt = _gate_run(repo, "live", "running")
    res = apply(repo, ["live"], by="t", lock_reader=no_lock)
    assert res.removed == []
    assert res.skipped == [("live", "RUN_NOT_FINISHED")]
    assert wt.exists()


def test_lock_holder_and_keep_are_blocked(repo: Path) -> None:
    _gate_run(repo, "a", "completed")
    _gate_run(repo, "b", "completed")
    (repo / "outputs" / "runs" / "b" / "KEEP").touch()
    lock = LockStatus(who="x", job="j", run_id="a", host_pid=1, started_at="t", alive=True)
    got = {i.run_id: i.blocked_by for i in plan(repo, lock_reader=lambda _p: lock)}
    assert got == {"a": "GPU_LOCK_HOLDER", "b": "PINNED_OR_KEEP"}


def test_apply_removes_via_git_and_keeps_outputs(repo: Path) -> None:
    wt = _gate_run(repo, "done", "completed")
    (repo / "outputs" / "runs" / "done" / "metrics.jsonl").write_text("{}\n")
    res = apply(repo, ["done"], by="tester", lock_reader=no_lock)
    assert [i.run_id for i in res.removed] == ["done"]
    assert not wt.exists()
    code, listed = git_runner(repo)(["worktree", "list", "--porcelain"])
    assert code == 0
    assert "done" not in listed  # no orphan registration left behind
    assert (repo / "outputs" / "runs" / "done" / "metrics.jsonl").exists()
    log = (repo / "outputs" / "system" / "cleanup.jsonl").read_text()
    assert '"worktree_removed"' in log


def test_dirty_worktree_is_not_forced(repo: Path) -> None:
    wt = _gate_run(repo, "done", "completed")
    (wt / "pyproject.toml").write_text("changed\n")
    res = apply(repo, ["done"], by="t", lock_reader=no_lock)
    assert res.removed == []
    assert [r for r, _msg in res.failed] == ["done"]
    assert wt.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["worktree", "remove", "--force", "x"],
        ["worktree", "remove", "-f", "x"],
        ["worktree", "add", "x"],
        ["reset", "--hard"],
        ["clean", "-fdx"],
        ["worktree", "list"],
        [],
    ],
)
def test_git_runner_refuses_anything_but_list_remove_prune(repo: Path, argv: list[str]) -> None:
    with pytest.raises(ValueError, match="list/remove/prune"):
        git_runner(repo)(argv)


def test_git_env_cannot_redirect_to_another_repo(
    repo: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path_factory.mktemp("other")
    _git(other, "init", "-q")
    _gate_run(repo, "done", "completed")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert [i.run_id for i in plan(repo, lock_reader=no_lock)] == ["done"]


def test_apply_refuses_wrong_project_root(repo: Path) -> None:
    wt = _gate_run(repo, "done", "completed")
    (repo / "pyproject.toml").write_text('[project]\nname = "someone-else"\n')
    with pytest.raises(ValueError, match="expected 'vlm-pipeline'"):
        apply(repo, ["done"], by="t", lock_reader=no_lock)
    assert wt.exists()
