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
        "entrypoint": None,
        "group_add": [],
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


def test_entrypoint_override_goes_before_the_image(tmp_path: Path) -> None:
    _, _, args = _job(tmp_path, allow_unapproved_image=True, entrypoint="python3")
    i = args.index("vlm-vllm:0.31.0")
    assert args[i - 2 : i] == ["--entrypoint", "python3"]
    assert args[i + 1 :] == ["python", "x.py"]  # argv follows the image unchanged
    _, _, default = _job(tmp_path / "b")
    assert "--entrypoint" not in default


@pytest.mark.parametrize("bad", ["-v /:/host", "--privileged", "python3 -c x", "", "a;b"])
def test_entrypoint_must_be_one_plain_token(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="invalid entrypoint"):
        _job(tmp_path, entrypoint=bad)


# ------------------------------------------------------------------------- --max-used-pct (80% rule)


def _main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str) -> tuple[int | None, dict[str, object]]:
    """Run jobrun_cli.main with run_job captured (no docker); returns the exit and run_job's keyword args."""
    import vlm_pipeline.common.jobrun_cli as cli
    from vlm_pipeline.common.runinfo import RunStatus

    captured: dict[str, object] = {}

    def fake_run_job(job: object, manifest: object, run_dir: Path, **kw: object) -> object:
        captured.update(kw)
        return argparse.Namespace(status=RunStatus.COMPLETED, error_code=None)

    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "run_job", fake_run_job)
    argv = ["--image", "vlm-automodel:26.08-5", "--tag", "t", "--est-peak-gb", "8", *flags, "--", "python", "x.py"]
    return cli.main(argv), captured


def test_max_free_pct_tightens_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, kw = _main(tmp_path, monkeypatch, "--max-free-pct", "70")
    assert code in (0, None)
    thresholds = kw["mem_thresholds"]
    assert thresholds.max_free_frac == pytest.approx(0.70)  # type: ignore[attr-defined]
    assert thresholds.warn_free_frac < thresholds.max_free_frac  # type: ignore[attr-defined]


def test_default_ceiling_is_80(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, kw = _main(tmp_path, monkeypatch)
    assert kw["mem_thresholds"].max_free_frac == pytest.approx(0.80)  # type: ignore[attr-defined]


@pytest.mark.parametrize("pct", ["90", "80.5", "0", "nan"])
def test_a_looser_ceiling_is_refused_before_any_run_dir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pct: str
) -> None:
    from vlm_pipeline.common.jobrun import ThresholdTooLooseError

    with pytest.raises((ThresholdTooLooseError, SystemExit)):
        _main(tmp_path, monkeypatch, "--max-free-pct", pct)
    runs = tmp_path / "outputs" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []  # no orphan run.json stuck in "running"


def test_group_add_reaches_the_job_only_for_its_allowed_image(tmp_path: Path) -> None:
    """research's NeMo-RL probe: --group-add 0 for vlm-nemo-rl only (security 2026-10-14)."""
    nemo_rl = "sha256:4c740d252ef90651621224005623546b4931582fa2eeb9ce816ba9d7576a4add"  # run_job pins the tag to it
    _root, _run, argv = _job(tmp_path, image=nemo_rl, group_add=[0])
    assert argv[argv.index("--group-add") + 1] == "0"
    with pytest.raises(ValueError, match="may not join group"):
        _job(tmp_path / "tag", image="vlm-nemo-rl:0.7.0-1", group_add=[0])  # a tag alone is never trusted
    with pytest.raises(ValueError, match="may not join group"):
        _job(tmp_path / "other", group_add=[0])  # vlm-vllm: never
