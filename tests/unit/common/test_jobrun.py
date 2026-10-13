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


def test_rw_dirs_are_the_only_writable_mounts_and_must_be_under_outputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run = root / "outputs" / "graph" / "runs" / "try-1"
    run.mkdir(parents=True)
    args = docker_args(_job(read_only_project=True, rw_dirs=(run,)), root)
    assert f"{root}:/workspace:ro" in args
    assert f"{run.resolve()}:/workspace/outputs/graph/runs/try-1" in args
    for outside in (root / "src", root / "outputs" / ".." / "src", root, root / "outputs" / "system" / "x"):
        with pytest.raises(ValueError, match="must be inside"):
            docker_args(_job(read_only_project=True, rw_dirs=(outside,)), root)


@pytest.mark.parametrize(
    "key",
    ["HF_TOKEN", "NGC_API_KEY", "WANDB_API_KEY", "OPENAI_API_KEY", "HUGGINGFACE_TOKEN", "NVIDIA_API_KEY",
     "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD", "GCP_CREDENTIALS"],
)  # fmt: skip
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
    with pytest.raises(ThresholdTooLooseError):
        tightened_thresholds(float("nan"))
    with pytest.raises(ThresholdTooLooseError) as ei:
        tightened_thresholds(6.0)
    assert ei.value.code == "CFG-005"
    assert ei.value.params["param"] == "abort_gb"


def test_lock_token_and_normal_env_still_pass() -> None:
    token = "t"  # noqa: S105 - dummy
    env = {"VLM_EXPECTED_TRAINABLE_PCT": "29.4:31.4", "TOKENIZERS_PARALLELISM": "false", "MAX_JOBS": "4",
           "HF_HUB_ENABLE_HF_TRANSFER": "0"}  # fmt: skip
    args = docker_args(_job(env=env), Path("/repo"), lock_token=token)
    assert "TOKENIZERS_PARALLELISM=false" in args
    assert "VLM_GPU_LOCK_TOKEN=t" in args
    assert "VLM_EXPECTED_TRAINABLE_PCT=29.4:31.4" in args


def test_graceful_stop_line_detected(tmp_path: Path) -> None:
    from vlm_pipeline.common.jobrun import _stopped_by_signal

    log = tmp_path / "stdout.log"
    log.write_text("step 9\n2026-10-12 | INFO | root | Received signal 15, initiating graceful stop\n")
    assert _stopped_by_signal(log)
    log.write_text("step 59\ndone\n")
    assert not _stopped_by_signal(log)
    assert not _stopped_by_signal(tmp_path / "missing.log")


def test_git_state_dirty_is_scoped_to_dirs(tmp_path: Path) -> None:
    import subprocess

    from vlm_pipeline.common.jobrun import git_state

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)  # noqa: S603,S607

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", "src/a.py")
    git("commit", "-q", "-m", "init")
    (tmp_path / "scripts" / "new.py").write_text("")  # untracked, outside src/
    assert git_state(tmp_path)[1] is True
    assert git_state(tmp_path, ("src",))[1] is False
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert git_state(tmp_path, ("src",))[1] is True


def test_engine_warning_fixtures_match() -> None:
    """Each code has a real log line sample (tests/fixtures/engine_warnings/<code>.txt) that its pattern matches."""
    from vlm_pipeline.common.jobrun import ENGINE_WARNINGS, _engine_warnings

    fixtures = Path(__file__).parents[2] / "fixtures" / "engine_warnings"
    for code in ENGINE_WARNINGS:
        assert _engine_warnings(fixtures / f"{code}.txt") == [code], code
    assert _engine_warnings(Path("/nonexistent/stdout.log")) == []


def test_env_files_are_masked_in_the_project_mount(tmp_path: Path) -> None:
    """SEC-005: jobs run as the project owner's uid, so .env* (HF token) must be covered, ro mount or not."""
    for name in (".env", ".env.local", ".env.example"):
        (tmp_path / name).write_text("HF_TOKEN=hf_x\n")
    (tmp_path / "data").mkdir()
    for ro in (False, True):
        args = docker_args(_job(read_only_project=ro), tmp_path)
        masks = [args[i + 1] for i, a in enumerate(args) if a == "-v" and args[i + 1].startswith("/dev/null:")]
        assert masks == [f"/dev/null:/workspace/{n}:ro" for n in (".env", ".env.example", ".env.local")]
        project = args.index(f"{tmp_path}:/workspace" + (":ro" if ro else ""))
        assert all(args.index(m) > project for m in masks)  # later mounts shadow the project mount


def _probe_root(tmp_path: Path, verdict: str) -> Path:
    (tmp_path / "docker").mkdir()
    row = f"vlm-x | vlm-x:1 (local) | - | base | sha256:{'a' * 64} | 1GB | {verdict} | 2026-10-13 | probe\n"
    (tmp_path / "docker" / "IMAGES.lock").write_text("# header\n" + row)
    return tmp_path


@pytest.mark.parametrize("verdict", ["PROBE, not approved", "bao-security APPROVED-PENDING"])
def test_run_job_refuses_unapproved_images_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    """SEC-004b: the gate goes through run_job itself and is fail-closed (no preflight, no lock, no container)."""
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.images import ImageMismatchError
    from vlm_pipeline.common.runinfo import RunManifest

    root = _probe_root(tmp_path, verdict)
    monkeypatch.setattr(jobrun, "image_digest", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr("vlm_pipeline.common.preflight.preflight", lambda *a, **k: pytest.fail("preflight ran"))
    job = ContainerJob(name="r1", image="vlm-x:1", argv=("true",))
    manifest = RunManifest(run_id="r1", name="r1", stage="eval", task="eval")
    with pytest.raises(ImageMismatchError):
        jobrun.run_job(job, manifest, tmp_path / "run", project_root=root, who="t", est_peak_gb=1)


def _approved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ContainerJob, object]:
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.runinfo import RunManifest

    root = _probe_root(tmp_path, "bao-security APPROVED 2026-10-13")
    monkeypatch.setattr(jobrun, "image_digest", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr("vlm_pipeline.common.preflight.preflight", lambda *a, **k: pytest.fail("no preflight/lock"))
    return (
        root,
        ContainerJob(name="r1", image="vlm-x:1", argv=("true",)),
        RunManifest(run_id="r1", name="r1", stage="eval", task="eval"),
    )


def test_cancel_while_waiting_for_memory_starts_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.runinfo import read_manifest

    root, job, manifest = _approved_run(tmp_path, monkeypatch)
    seen: list[dict[str, float]] = []

    def fake_wait(est: float, *, on_wait: object, cancel: object, **kw: object) -> bool:
        on_wait({"need_gb": 30.0, "avail_gb": 12.0, "waited_s": 0.0})  # type: ignore[operator]
        return False  # cancelled

    def broken_ui(info: dict[str, float]) -> None:
        seen.append(info)
        raise RuntimeError("websocket closed")  # must not stop the wait

    monkeypatch.setattr("vlm_pipeline.common.preflight.wait_for_memory", fake_wait)
    final = jobrun.run_job(job, manifest, tmp_path / "run", project_root=root, who="t", est_peak_gb=30,  # type: ignore[arg-type]
                           on_wait=broken_ui)  # fmt: skip
    assert final.status == RunStatus.CANCELLED
    assert read_manifest(tmp_path / "run").status == RunStatus.CANCELLED
    assert (seen[0]["need_gb"], seen[0]["avail_gb"]) == (30.0, 12.0)
    assert "since" in seen[0]


def test_memory_wait_timeout_is_sys002_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.preflight import MemoryWaitTimeoutError
    from vlm_pipeline.common.runinfo import read_manifest

    root, job, manifest = _approved_run(tmp_path, monkeypatch)

    def timeout(*a: object, **k: object) -> bool:
        raise MemoryWaitTimeoutError("no memory", params={"minutes": 30})

    monkeypatch.setattr("vlm_pipeline.common.preflight.wait_for_memory", timeout)
    with pytest.raises(MemoryWaitTimeoutError) as ei:
        jobrun.run_job(job, manifest, tmp_path / "run", project_root=root, who="t", est_peak_gb=30)  # type: ignore[arg-type]
    assert ei.value.code == "SYS-002"
    recorded = read_manifest(tmp_path / "run")
    assert (recorded.status, recorded.error_code) == (RunStatus.FAILED, "SYS-002")


def test_wait_and_preflight_use_the_same_tightened_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tightened budget must keep the job WAITING, not let it through the wait and fail in preflight."""
    from vlm_pipeline.common import jobrun

    root, job, manifest = _approved_run(tmp_path, monkeypatch)
    tight = jobrun.tightened_thresholds(abort_gb=20.0)
    seen: dict[str, object] = {}

    def fake_wait(est: float, *, thresholds: object, **kw: object) -> bool:
        seen["wait"] = thresholds
        return False

    monkeypatch.setattr("vlm_pipeline.common.preflight.wait_for_memory", fake_wait)
    jobrun.run_job(job, manifest, tmp_path / "run", project_root=root, who="t", est_peak_gb=10,  # type: ignore[arg-type]
                   mem_thresholds=tight)  # fmt: skip
    assert seen["wait"] is tight


def test_compile_caches_are_kept_per_image_and_follow_the_writable_dir(tmp_path: Path) -> None:
    from vlm_pipeline.common.jobrun import compile_cache_env

    run = tmp_path / "outputs" / "graph" / "runs" / "n1"
    rw_env = compile_cache_env(_job(), tmp_path, "sha256:" + "ab" * 32)
    assert rw_env == {
        "TORCH_EXTENSIONS_DIR": "/workspace/.cache/torch_extensions/abababababababab",
        "TORCHINDUCTOR_CACHE_DIR": "/workspace/.cache/inductor/abababababababab",
        "MAX_JOBS": "2",
    }
    ro_env = compile_cache_env(_job(read_only_project=True, rw_dirs=(run,)), tmp_path, "sha256:" + "ef" * 32)
    assert ro_env["TORCH_EXTENSIONS_DIR"] == "/workspace/outputs/graph/runs/n1/.cache/torch_extensions/efefefefefefefef"
    unknown = compile_cache_env(_job(), tmp_path, None)["TORCH_EXTENSIONS_DIR"]
    assert not unknown.startswith("/workspace")  # no image id: rebuilt per job, never a shared "unknown" cache
    no_rw = compile_cache_env(_job(read_only_project=True), tmp_path, "sha256:" + "ef" * 32)["TORCH_EXTENSIONS_DIR"]
    assert not no_rw.startswith("/workspace")  # nothing writable in the project: per-job scratch
    args = docker_args(_job(env={"MAX_JOBS": "4"}), tmp_path, image_id="sha256:" + "cd" * 32)
    assert "MAX_JOBS=4" in args  # a job's own env wins
    assert "TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor/cdcdcdcdcdcdcdcd" in args
