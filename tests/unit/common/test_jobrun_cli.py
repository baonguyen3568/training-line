"""jobrun_cli: an image bao-security has not approved runs with a read-only project unless explicitly opted out."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from vlm_pipeline.common.jobrun import docker_args
from vlm_pipeline.common.jobrun_cli import build_job


def _args(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "image": "vlm-vllm:0.31.0",
        "no_gpu": False,
        "read_only_project": False,
        "rw_dir": [],
        "allow_unapproved_image": False,
        "unapproved_image_rw": False,
    }
    return argparse.Namespace(**{**base, **kw})


def _job(tmp_path: Path, **kw: object) -> tuple[Path, Path, list[str]]:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    job = build_job(_args(**kw), root, "r1", run_dir, ("python", "x.py"))
    return root, run_dir, docker_args(job, root)


def test_unapproved_image_is_read_only_with_the_run_dir_writable(tmp_path: Path) -> None:
    root, run_dir, args = _job(tmp_path, allow_unapproved_image=True, rw_dir=[Path("outputs/probe/x")])
    assert f"{root}:/workspace:ro" in args
    assert f"{run_dir.resolve()}:/workspace/outputs/runs/r1" in args
    assert f"{(root / 'outputs/probe/x').resolve()}:/workspace/outputs/probe/x" in args


def test_opt_out_keeps_the_project_writable(tmp_path: Path) -> None:
    root, _, args = _job(tmp_path, allow_unapproved_image=True, unapproved_image_rw=True)
    assert f"{root}:/workspace" in args
    assert f"{root}:/workspace:ro" not in args


def test_approved_image_default_is_unchanged_and_read_only_is_opt_in(tmp_path: Path) -> None:
    root, _, args = _job(tmp_path)
    assert f"{root}:/workspace" in args
    root2, _, args2 = _job(tmp_path / "b", read_only_project=True)
    assert f"{root2}:/workspace:ro" in args2


@pytest.mark.parametrize("bad", ["../../outside", "outputs/system/x", "src/x"])
def test_rejected_rw_dir_is_never_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    from vlm_pipeline.common import jobrun_cli

    root = tmp_path / "a" / "b" / "repo"
    root.mkdir(parents=True)
    monkeypatch.setattr(jobrun_cli, "project_root", lambda: root)
    monkeypatch.setattr(jobrun_cli, "run_job", lambda *a, **kw: pytest.fail("job must not start"))
    argv = ["--image", "vlm-vllm:0.31.0", "--tag", "t", "--est-peak-gb", "1", "--allow-unapproved-image",
            "--rw-dir", bad, "--", "true"]  # fmt: skip
    with pytest.raises(SystemExit):
        jobrun_cli.main(argv)
    assert not (root / bad).exists()
    assert not (root / bad).resolve().exists()


def test_validate_rw_dirs_accepts_outputs_subdirs_only(tmp_path: Path) -> None:
    from vlm_pipeline.common.jobrun import validate_rw_dirs

    root = tmp_path / "repo"
    assert validate_rw_dirs(root, [root / "outputs" / "probe" / "x"]) == [(root / "outputs/probe/x").resolve()]
    for bad in (root / "outputs", root / "outputs" / "system", Path("/abs/x")):
        with pytest.raises(ValueError, match="must be inside"):
            validate_rw_dirs(root, [bad])
