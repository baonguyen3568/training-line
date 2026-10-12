from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.jobrun import ContainerJob, ExitInfo, classify_exit, docker_args
from vlm_pipeline.common.runinfo import RunStatus


def _job(**kw: object) -> ContainerJob:
    return ContainerJob(
        name="r1", image="vlm-automodel:26.08-1", argv=("automodel", "cfg.yaml"), uid=1000, gid=1000, **kw
    )  # type: ignore[arg-type]


def test_docker_args_baseline() -> None:
    token = "tok"  # noqa: S105 - dummy lock token
    args = docker_args(_job(), Path("/repo"), lock_token=token)
    joined = " ".join(args)
    for must in ("--network none", "--cap-drop ALL", "--security-opt no-new-privileges", "--user 1000:1000",
                 "--label project=vlm-training", "--name vlm-r1", "-v /repo:/workspace", "--gpus all",
                 "HF_HUB_OFFLINE=1", "VLM_GPU_LOCK_TOKEN=tok"):  # fmt: skip
        assert must in joined
    assert "--rm" not in args
    assert "--privileged" not in joined
    assert "--network host" not in joined
    assert args[-3:] == ["vlm-automodel:26.08-1", "automodel", "cfg.yaml"]


def test_no_gpu_and_read_only_mount() -> None:
    args = docker_args(_job(gpu=False, read_only_project=True), Path("/repo"))
    assert "--gpus" not in args
    assert "/repo:/workspace:ro" in args


@pytest.mark.parametrize("key", ["HF_TOKEN", "NGC_API_KEY", "WANDB_API_KEY"])
def test_secrets_never_reach_containers(key: str) -> None:
    with pytest.raises(ValueError, match="secret-like"):
        docker_args(_job(env={key: "x"}), Path("/repo"))


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (ExitInfo(0, False), (RunStatus.COMPLETED, None)),
        (ExitInfo(75, False), (RunStatus.INTERRUPTED, "MEM-002")),
        (ExitInfo(137, True), (RunStatus.FAILED, "MEM-003")),
        (ExitInfo(137, False), (RunStatus.FAILED, "SYS-003")),
        (ExitInfo(139, False), (RunStatus.FAILED, "SYS-003")),
        (ExitInfo(137, True, cancelled_by_executor=True), (RunStatus.CANCELLED, None)),
        (ExitInfo(143, False, cancelled_by_executor=True), (RunStatus.CANCELLED, None)),
    ],
)
def test_classify_exit(info: ExitInfo, expected: tuple[RunStatus, str | None]) -> None:
    assert classify_exit(info) == expected


def test_git_state_on_this_repo() -> None:
    from vlm_pipeline.common.jobrun import git_state
    from vlm_pipeline.common.paths import project_root

    commit, dirty = git_state(project_root())
    assert commit is not None
    assert len(commit) == 40
    assert isinstance(dirty, bool)


def test_git_state_outside_repo(tmp_path: Path) -> None:
    from vlm_pipeline.common.jobrun import git_state

    assert git_state(tmp_path) == (None, None)


def test_worktree_overlays_code_read_only() -> None:
    args = docker_args(_job(code_root=Path("/repo/.worktrees/r1")), Path("/repo"))
    joined = " ".join(args)
    assert "-v /repo:/workspace" in joined
    for d in ("src", "configs", "scripts", "docker"):
        assert f"/repo/.worktrees/r1/{d}:/workspace/{d}:ro" in joined


def test_thresholds_can_only_be_tightened() -> None:
    from vlm_pipeline.common.jobrun import ThresholdTooLooseError, tightened_thresholds
    from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS

    assert tightened_thresholds(None) == DEFAULT_THRESHOLDS
    th = tightened_thresholds(10.0)
    assert th.abort_gb == 10.0
    assert th.warn_gb > th.abort_gb > th.emergency_gb
    assert tightened_thresholds(8.0).abort_gb == 8.0
    with pytest.raises(ThresholdTooLooseError) as ei:
        tightened_thresholds(6.0)
    assert ei.value.code == "CFG-005"
    assert ei.value.params["param"] == "abort_gb"
