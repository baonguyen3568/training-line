from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        (ExitInfo(1, False, cuda_oom_in_log=True), (RunStatus.FAILED, "MEM-003")),  # evidence in the log
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


def test_a_read_only_job_never_gets_a_triton_cache_in_the_read_only_project(tmp_path: Path) -> None:
    # 2026-10-13 quantize GPU run: Qwen3.5 Triton autotune hit BASE_ENV's /workspace/.cache/triton (mounted :ro).
    from vlm_pipeline.common.jobrun import BASE_ENV, compile_cache_env

    run = tmp_path / "outputs" / "graph" / "runs" / "n1"
    for job, image in (
        (_job(read_only_project=True, rw_dirs=(run,)), "sha256:" + "ef" * 32),
        (_job(read_only_project=True), "sha256:" + "ef" * 32),
        (_job(read_only_project=True), None),
    ):
        triton = {**BASE_ENV, **compile_cache_env(job, tmp_path, image)}["TRITON_CACHE_DIR"]
        assert triton.startswith(("/workspace/outputs/", "/tmp/")), triton  # noqa: S108
    assert "TRITON_CACHE_DIR" not in compile_cache_env(_job(), tmp_path, "sha256:" + "ab" * 32)  # writable: unchanged


def test_a_run_id_gets_its_job_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P017: run_job claims the run id right before the container; the same id never gets a second job."""
    import threading
    from types import SimpleNamespace

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.jobrun import ExitInfo
    from vlm_pipeline.common.run_claims import RunDirClaimedError, is_claimed
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus

    root = _probe_root(tmp_path, "bao-security APPROVED 2026-10-13")
    monkeypatch.setattr(jobrun, "image_digest", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr(
        "vlm_pipeline.common.preflight.preflight", lambda *a, **k: SimpleNamespace(ok=True, findings=())
    )
    monkeypatch.setattr(jobrun, "_wait_for_memory", lambda *a, **k: None)
    monkeypatch.setattr(jobrun, "git_state", lambda _root: ("c0ffee", False))
    monkeypatch.setattr(jobrun, "read_meminfo", lambda: SimpleNamespace(mem_available_gb=40.0))
    guard = SimpleNamespace(should_stop=threading.Event(), min_available_gb=30.0, samples=3, fast_fall_killed=False,
                            thresholds=SimpleNamespace(abort_gb=8.0))  # fmt: skip
    started: list[str] = []
    monkeypatch.setattr(jobrun, "run_guarded", lambda launch, *a, **k: (launch(), guard))

    class Launcher:
        def run(self, job: ContainerJob, *a: object, **k: object) -> int:
            started.append(job.name)
            return 0

        def state(self, name: str) -> ExitInfo:
            return ExitInfo(exit_code=0, oom_killed=False)

        def stop(self, *a: object) -> None: ...

        def kill(self, *a: object) -> None: ...

        def remove(self, *a: object) -> None: ...

    def run() -> RunManifest:
        return jobrun.run_job(
            ContainerJob(name="n4-0123456789abcdef", image="vlm-x:1", argv=("true",)),
            RunManifest(run_id="n4-0123456789abcdef", name="t", stage="train", task="sft"),
            root / "outputs" / "graph" / "runs" / "n4-0123456789abcdef", project_root=root, who="t", est_peak_gb=1,
            lock_path=tmp_path / "GPU_LOCK", launcher=Launcher(),  # type: ignore[arg-type]
        )  # fmt: skip

    assert run().status == RunStatus.COMPLETED
    assert is_claimed(root, "n4-0123456789abcdef")
    run_json = root / "outputs" / "graph" / "runs" / "n4-0123456789abcdef" / "run.json"
    before = run_json.read_bytes()
    with pytest.raises(RunDirClaimedError):  # B1: refused before anything is written
        run()
    monkeypatch.setattr(jobrun, "is_claimed", lambda *_a: False)  # lost race: only the atomic claim catches it
    with pytest.raises(RunDirClaimedError):
        run()
    assert run_json.read_bytes() == before  # the first job's history is never rewritten
    assert started == ["n4-0123456789abcdef"]  # no second job reached the container


def _narrow_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "proj"
    for d in ("src/vlm_pipeline", "configs/train"):
        (root / d).mkdir(parents=True)
    run = root / "outputs" / "graph" / "runs" / "t-0123456789abcdef"
    run.mkdir(parents=True)
    repo = root / ".cache" / "hf" / "hub" / "models--org--m"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "abc").write_bytes(b"w")
    (repo / "snapshots" / "rev").mkdir(parents=True)
    (repo / "snapshots" / "rev" / "model.safetensors").symlink_to("../../blobs/abc")  # how HF lays out its cache
    data = root / "data" / "mine"
    data.mkdir(parents=True)
    (data / "labels.jsonl").write_text("{}\n")
    (root / ".env.local").write_text("HF_TOKEN=x\n")
    return root, run, repo, data


def _mounts(argv: list[str]) -> list[tuple[str, str, str]]:
    """(source, target, mode) of every -v."""
    out = []
    for i, a in enumerate(argv):
        if a == "-v":
            src, dst, *mode = argv[i + 1].split(":")
            out.append((src, dst, mode[0] if mode else "rw"))
    return out


def test_a_narrowed_job_sees_only_what_it_needs(tmp_path: Path) -> None:
    """P017 step 6 (bao-security S6): src/configs ro, ONE rw dir (its run dir), inputs ro; no project, no secrets."""
    root, run, repo, data = _narrow_project(tmp_path)
    job = ContainerJob(name="t", image="img", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(repo, data))
    argv = docker_args(job, root)
    mounts = _mounts(argv)
    sources = [Path(s) for s, _, _ in mounts]
    assert not any(root == s or root.is_relative_to(s) for s in sources)  # never the project or one of its parents
    assert not any(".env" in s.name or "system" in s.parts for s in sources)
    assert [s for s, _, m in mounts if m != "ro"] == [str(run)]  # exactly one writable mount: its own run dir
    assert (str(repo), "/workspace/.cache/hf/hub/models--org--m", "ro") in mounts
    assert (str(data), "/workspace/data/mine", "ro") in mounts
    assert {d for _, d, m in mounts if m == "ro"} >= {"/workspace/src", "/workspace/configs"}
    assert "--tmpfs" in argv
    assert "TRITON_CACHE_DIR=/workspace/outputs/graph/runs/t-0123456789abcdef/.cache/triton" in argv  # never shared


@pytest.mark.parametrize(
    "plant",
    ["absolute_link", "link_out", "system_input", "env_input", "overlap", "outside", "link_in_path", "two_rw"],
)
def test_a_narrowed_job_refuses_unsafe_sources(tmp_path: Path, plant: str) -> None:
    root, run, repo, data = _narrow_project(tmp_path)
    inputs: tuple[Path, ...] = (repo,)
    rw: tuple[Path, ...] = (run,)
    if plant == "absolute_link":  # a job with rw on the project could have planted it
        (repo / "snapshots" / "rev" / "leak").symlink_to("/etc/passwd")
    elif plant == "link_out":
        (repo / "snapshots" / "rev" / "leak").symlink_to("../../../../../../data/mine")
    elif plant == "system_input":
        (root / "outputs" / "system").mkdir()
        inputs = (root / "outputs" / "system",)
    elif plant == "env_input":
        inputs = (root / ".env.local",)
    elif plant == "overlap":
        inputs = (run.parent,)  # a parent of the writable dir
    elif plant == "outside":
        inputs = (tmp_path,)
    elif plant == "link_in_path":
        (root / "data" / "alias").symlink_to(data)
        inputs = (root / "data" / "alias",)
    else:
        other = root / "outputs" / "graph" / "runs" / "u-0123456789abcdef"
        other.mkdir()
        rw = (run, other)
    job = ContainerJob(name="t", image="img", argv=("true",), narrow=True, rw_dirs=rw, ro_inputs=inputs)
    with pytest.raises(ValueError, match=r"(?i)input|writable|symlink|outside|not a"):  # UnsafeMountSourceError too
        docker_args(job, root)


def _gpu_wait_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdicts: list[object]) -> tuple[Any, Any]:
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.runinfo import RunManifest

    monkeypatch.setattr(jobrun, "GPU_POLL_S", 0.01)
    calls = iter(verdicts)
    monkeypatch.setattr("vlm_pipeline.common.preflight.preflight", lambda *a, **k: next(calls))
    manifest = RunManifest(run_id="n4-0123456789abcdef", name="t", stage="train", task="sft")
    run_dir = tmp_path / "outputs" / "graph" / "runs" / "n4-0123456789abcdef"
    run_dir.mkdir(parents=True)
    return (SimpleNamespace, jobrun), (manifest, run_dir)


def _busy(state: str = "busy") -> object:
    from vlm_pipeline.common.errors import ExitCode
    from vlm_pipeline.common.preflight import Finding

    f = Finding("SYS-001", ExitCode.LOCK_BUSY, "GPU busy", {"who": "x", "job": "y", "since": "s", "state": state})
    return SimpleNamespace(ok=False, findings=(f,))


def test_a_busy_gpu_is_waited_for_not_an_unknown_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """research R2: Run while another job trains -> the job queues (waiting_gpu), then takes the lock."""
    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(
        tmp_path, monkeypatch, [_busy(), _busy(), SimpleNamespace(ok=True, findings=())]
    )
    seen: list[dict[str, float]] = []
    handle = jobrun._wait_for_gpu(
        manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
        cancel=None, on_wait=seen.append, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS,
    )  # fmt: skip
    assert handle.info.run_id == "n4-0123456789abcdef"
    assert len(seen) == 2
    assert all(s["gpu_busy"] == 1.0 for s in seen)


def _short(memory: str | None = None) -> object:
    from vlm_pipeline.common.errors import ExitCode
    from vlm_pipeline.common.preflight import Finding

    params: dict[str, object] = {"need_gb": 32.9, "avail_gb": 31.9, "est_peak_gb": 24.9, "max_free_pct": 80}
    if memory:
        params["memory"] = memory
    return SimpleNamespace(ok=False, findings=(Finding("MEM-001", ExitCode.NO_MEM, "short", params),))


def test_memory_short_when_the_lock_frees_up_is_waited_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GB10 2026-10-13 (run ...-q08-bf16-kv8-b353): the job queued for the lock 33 min, then FAILED MEM-001 because
    the shared host's free memory had changed meanwhile. Queue, never fail (PM): it waits and then runs."""
    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(
        tmp_path, monkeypatch, [_busy(), _short(), _short("vram"), SimpleNamespace(ok=True, findings=())]
    )
    seen: list[dict[str, float]] = []
    handle = jobrun._wait_for_gpu(
        manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
        cancel=None, on_wait=seen.append, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS,
    )  # fmt: skip
    assert handle.info.run_id == "n4-0123456789abcdef"
    assert seen[0]["gpu_busy"] == 1.0
    assert (seen[1]["need_gb"], seen[1]["avail_gb"]) == (32.9, 31.9)
    assert "gpu_busy" not in seen[1]
    assert seen[2]["vram_need_gb"] == 32.9


def test_a_memory_shortage_past_the_wait_is_sys002(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common.runinfo import read_manifest

    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(tmp_path, monkeypatch, [_short()] * 5)
    with pytest.raises(jobrun.PreflightRefusedError) as err:
        jobrun._wait_for_gpu(manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
                             cancel=None, on_wait=None, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS,
                             memory_wait_s=0.0)  # fmt: skip
    assert err.value.code == "SYS-002"
    assert err.value.params == {"minutes": 1, "waiting_for": "memory", "need_gb": 32.9, "avail_gb": 31.9}
    assert read_manifest(run_dir).error_code == "SYS-002"


def test_a_shortage_behind_a_busy_lock_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Behind a long training run the memory is mostly the holder's: only a shortage with a FREE GPU times out."""
    both = SimpleNamespace(ok=False, findings=(*_busy().findings, *_short().findings))  # type: ignore[attr-defined]
    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(
        tmp_path, monkeypatch, [both, both, SimpleNamespace(ok=True, findings=())]
    )
    handle = jobrun._wait_for_gpu(
        manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
        cancel=None, on_wait=None, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS, memory_wait_s=0.0,
    )  # fmt: skip
    assert handle.info.run_id == "n4-0123456789abcdef"


@pytest.mark.parametrize("state", ["stale", "corrupt"])
def test_a_stale_lock_fails_at_once_with_its_own_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    from vlm_pipeline.common.runinfo import read_manifest

    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(tmp_path, monkeypatch, [_busy(state)])
    with pytest.raises(jobrun.PreflightRefusedError) as err:
        jobrun._wait_for_gpu(manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
                             cancel=None, on_wait=None, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS)  # fmt: skip
    assert err.value.code == "SYS-001"
    assert err.value.params["state"] == state
    assert (read_manifest(run_dir).status, read_manifest(run_dir).error_code) == ("failed", "SYS-001")


def test_cancel_while_queued_for_the_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    (_, jobrun), (manifest, run_dir) = _gpu_wait_setup(tmp_path, monkeypatch, [_busy()] * 5)
    cancel = threading.Event()
    cancel.set()
    out = jobrun._wait_for_gpu(
        manifest, run_dir, tmp_path / "GPU_LOCK", 1.0, 0.0, who="t", project_root=tmp_path,
        cancel=cancel, on_wait=None, timeout_s=60, thresholds=jobrun.DEFAULT_THRESHOLDS,
    )  # fmt: skip
    assert out.status == "cancelled"
    assert not (tmp_path / "GPU_LOCK").exists()


def test_hf_links_into_the_shared_blob_store_mount_only_those_blobs(tmp_path: Path) -> None:
    """Real HF cache: repo blobs link to hub/blobs/<xx>/<sha>; a prepared copy links into the repo. Only the files
    those chains pass through are added (never the whole store: other repos' blobs stay unseen)."""
    root, run, _repo, _data = _narrow_project(tmp_path)
    hub = root / ".cache" / "hf" / "hub"
    store = hub / "blobs" / "ab"
    store.mkdir(parents=True)
    (store / "abcd").write_bytes(b"w")
    (store / "other").write_bytes(b"not for this job")
    repo = hub / "models--org--p"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "sha1").symlink_to("../../blobs/ab/abcd")
    (repo / "snapshots" / "r").mkdir(parents=True)
    (repo / "snapshots" / "r" / "w.safetensors").symlink_to("../../blobs/sha1")
    prepared = root / ".cache" / "prepared" / "p_1" / "model"
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    (prepared / "w.safetensors").symlink_to("../../../hf/hub/models--org--p/blobs/sha1")
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(prepared,))
    mounts = {(s, m) for s, _, m in _mounts(docker_args(job, root))}
    assert (str(repo), "ro") in mounts
    assert (str(store / "abcd"), "ro") in mounts
    assert not any(s in (str(store), str(store / "other"), str(hub)) for s, _ in mounts)


@pytest.mark.parametrize(
    "rel",
    [
        ".cache",
        ".cache/hf",
        ".cache/hf/hub",
        "outputs/runs",
        "outputs/graph/runs",
        "data",
        "configs/../.team",
        "src",
        "uploads",
    ],
)
def test_only_allowed_input_roots_are_mounted(tmp_path: Path, rel: str) -> None:
    """bao-security S6-1: an allowlist of input roots, never a root itself (caches, other runs, the HF login token)."""
    root, run, _repo, _data = _narrow_project(tmp_path)
    target = root / rel
    target.mkdir(parents=True, exist_ok=True)
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(target,))
    with pytest.raises(ValueError, match=r"allowed input root|overlaps"):
        docker_args(job, root)


@pytest.mark.parametrize(
    "name",
    [
        "token",
        "stored_tokens",
        ".env.local",
        "id_ed25519",
        "server.pem",
        "tls.key",
        ".netrc",
        ".git-credentials",
        "credentials-gcp.json",
    ],
)
def test_an_input_holding_a_credential_is_refused(tmp_path: Path, name: str) -> None:
    root, run, _repo, data = _narrow_project(tmp_path)
    (data / name).write_text("x")
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(data,))
    with pytest.raises(ValueError, match=r"credential|secrets"):
        docker_args(job, root)


def test_a_job_cache_under_a_run_is_never_an_input(tmp_path: Path) -> None:
    """security nit (a): another job's .so / triton cache in its run dir is never mounted into the next job."""
    root, run, _repo, _data = _narrow_project(tmp_path)
    other = root / "outputs" / "graph" / "runs" / "u-0123456789abcdef" / ".cache" / "torch_extensions"
    other.mkdir(parents=True)
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(other,))
    with pytest.raises(ValueError, match="job cache"):
        docker_args(job, root)


def test_input_problem_matches_the_mount_rules(tmp_path: Path) -> None:
    """fixcode: callers that build inputs refuse early with the same rules narrow_mounts applies."""
    from vlm_pipeline.common.jobrun import input_problem

    root, _run, repo, data = _narrow_project(tmp_path)
    probe = root / "outputs" / "probe" / "inputs"
    probe.mkdir(parents=True)
    assert input_problem(root, repo) is None
    assert input_problem(root, data) is None
    assert "allowed input root" in (input_problem(root, probe) or "")
    assert "outside the project" in (input_problem(root, tmp_path) or "")


def test_a_refused_input_fails_before_the_run_dir_is_written(tmp_path: Path) -> None:
    """fixcode: SEC-001 naming the input, not SYS-000 after run.json exists (no claim, no image lookup)."""
    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.jobrun import JobInputRefusedError
    from vlm_pipeline.common.run_claims import is_claimed
    from vlm_pipeline.common.runinfo import RunManifest

    root, run, _repo, _data = _narrow_project(tmp_path)
    probe = root / "outputs" / "probe" / "inputs"
    probe.mkdir(parents=True)
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(probe,))
    with pytest.raises(JobInputRefusedError) as err:
        jobrun.run_job(
            job, RunManifest(run_id=run.name, name="t", stage="train", task="sft"), run, project_root=root,
            who="t", est_peak_gb=1, lock_path=tmp_path / "GPU_LOCK",
        )  # fmt: skip
    assert err.value.code == "SEC-001"
    assert err.value.params == {"path": "outputs/probe/inputs"}
    assert not (run / "run.json").exists()
    assert not is_claimed(root, run.name)


def test_vlm_code_root_runs_a_clean_worktrees_code_on_the_projects_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """research/E2E: src + configs from a worktree of HEAD; outputs, caches and the lock stay the project's."""
    import shutil

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.jobrun import CODE_ROOT_ENV, with_code_root

    root, run, _repo, data = _narrow_project(tmp_path)
    (root / "pyproject.toml").write_text('name = "vlm-pipeline"\n')
    wt = root / ".cache" / "wt-e2e"
    for d in ("src/vlm_pipeline", "configs/train"):
        (wt / d).mkdir(parents=True)
    (wt / "pyproject.toml").write_text('name = "vlm-pipeline"\n')
    (wt / ".git").write_text("gitdir: /x/.git/worktrees/wt-e2e\n")  # a worktree's .git is a file
    monkeypatch.setattr(jobrun, "git_state", lambda path: ("c0ffee", False))
    job = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(data,))
    assert with_code_root(job, root) is job  # unset: unchanged
    monkeypatch.setenv(CODE_ROOT_ENV, str(wt))
    moved = with_code_root(job, root)
    sources = {s: t for s, t, _m in _mounts(docker_args(moved, root))}
    assert sources[str(wt / "src")] == "/workspace/src"
    assert sources[str(run)] == f"/workspace/{run.relative_to(root)}"  # outputs stay in the project
    planted = run / "code"  # a run dir: a job could write pyproject.toml + src/ there (security C1)
    shutil.copytree(wt, planted)
    for bad in (tmp_path, planted, wt / "src"):
        monkeypatch.setenv(CODE_ROOT_ENV, str(bad))
        with pytest.raises(ValueError, match="git worktree checkout"):
            with_code_root(job, root)
    (wt / ".git").unlink()
    (wt / ".git").mkdir()  # a full clone, not a worktree
    monkeypatch.setenv(CODE_ROOT_ENV, str(wt))
    with pytest.raises(ValueError, match="git worktree checkout"):
        with_code_root(job, root)


def test_network_and_token_files_are_for_narrowed_download_jobs_only(tmp_path: Path) -> None:
    root, run, _repo, data = _narrow_project(tmp_path)
    secret = tmp_path / "tok"
    secret.write_text("hf_x")
    wide = ContainerJob(name="t", image="i", argv=("true",), egress=True)
    with pytest.raises(ValueError, match="narrowed download"):
        docker_args(wide, root)
    no_net = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), secret_file=secret)
    with pytest.raises(ValueError, match="only for a download job"):
        docker_args(no_net, root)
    link = tmp_path / "link"
    link.symlink_to(secret)
    auth = "t:" + "ab" * 16
    linked = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), egress=True,
                          secret_file=link, egress_auth=auth)  # fmt: skip
    with pytest.raises(ValueError, match="regular file"):
        docker_args(linked, root)
    no_auth = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), egress=True)
    with pytest.raises(ValueError, match="proxy credential"):
        docker_args(no_auth, root)
    ok = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), egress=True,
                      secret_file=secret, egress_auth=auth)  # fmt: skip
    argv = docker_args(ok, root)
    assert f"HTTPS_PROXY=http://{auth}@egress-proxy:3128" in argv
    assert argv[argv.index("--network") + 1] == "ntc-training_egress"
    plain = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(data,))
    argv = docker_args(plain, root)
    assert argv[argv.index("--network") + 1] == "none"
    assert not any(a.startswith(("HTTPS_PROXY", "HF_TOKEN_PATH")) for a in argv)


def test_a_cuda_allocation_failure_in_the_log_is_mem003_not_sys003(tmp_path: Path) -> None:
    """tester 2026-10-13 (run ...-2002-bench-r0-model-1f93): a CUDA OOM while loading showed 'stopped unexpectedly'."""
    from vlm_pipeline.common.jobrun import _cuda_oom_in_log

    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "stdout.log"
    log.write_text("    return t.to(\ntorch.AcceleratorError: CUDA error: out of memory\n")
    assert _cuda_oom_in_log(tmp_path)
    log.write_text("ValueError: bad config\n")
    assert not _cuda_oom_in_log(tmp_path)


def test_graph_classifies_a_cuda_oom_log_tail_as_mem003() -> None:
    from vlm_pipeline.graph.errors import ContainerExit, classify

    tail = 'File "torch/nn/modules/module.py", line 1369, in convert\ntorch.AcceleratorError: CUDA error: out of memory'
    out = classify(ContainerExit(1, log_tail=tail), cancelled=False, checkpoint_step=None)
    assert out.error is not None
    assert out.error.code == "MEM-003"
    other = classify(ContainerExit(1, log_tail="KeyError: 'x'"), cancelled=False, checkpoint_step=None)
    assert other.error is not None
    assert other.error.code == "SYS-003"


def test_a_job_is_never_budgeted_by_the_launchers_own_cgroup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM 2026-10-14: the compose app (4 GB cgroup) saw "3.9 GB available" and every job it launched waited
    forever. A run_job job runs in its OWN container: only host memory is its budget."""
    from vlm_pipeline.common import jobrun

    root, job, manifest = _approved_run(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def fake_wait(est: float, **kw: object) -> bool:
        seen["read_cgroup"] = kw.get("read_cgroup")
        return False  # cancelled: stop right after the memory wait

    monkeypatch.setattr("vlm_pipeline.common.preflight.wait_for_memory", fake_wait)
    jobrun.run_job(job, manifest, tmp_path / "run", project_root=root, who="t", est_peak_gb=2)  # type: ignore[arg-type]
    reader = seen["read_cgroup"]
    assert callable(reader)
    assert reader() is None


def test_preflight_can_leave_the_callers_cgroup_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common import preflight as pf
    from vlm_pipeline.common.memguard import MemSnapshot

    host = MemSnapshot(mem_total_gb=121.0, mem_available_gb=42.0, cached_gb=30.0, swap_free_gb=6.0)
    monkeypatch.setattr(pf, "read_meminfo", lambda: host)
    monkeypatch.setattr(pf, "read_cgroup_available_gb", lambda: 3.9)  # the app container's own 4 GB cap
    mem = [f.code for f in pf.preflight(2.0, lock_holder=lambda: None, gpus=[]).findings if f.code == "MEM-001"]
    assert mem == ["MEM-001"]  # the process itself would not fit in its own cgroup
    free = pf.preflight(2.0, lock_holder=lambda: None, gpus=[], own_cgroup=False).findings
    assert "MEM-001" not in [f.code for f in free]  # but a job in another container does


def test_an_app_capped_at_4gb_still_runs_a_2gb_job_when_the_host_has_40gb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PM 2026-10-14 regression: the REAL memory queue and preflight, only the readings faked. The app's own cgroup
    (4 GB cap, 3.9 GB headroom) must not hold back a job that runs in its own container."""
    import threading
    from types import SimpleNamespace

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common import preflight as pf
    from vlm_pipeline.common.jobrun import ExitInfo
    from vlm_pipeline.common.memguard import MemSnapshot
    from vlm_pipeline.common.runinfo import RunManifest, RunStatus

    host = MemSnapshot(mem_total_gb=121.0, mem_available_gb=40.0, cached_gb=30.0, swap_free_gb=6.0)
    monkeypatch.setattr(pf, "read_meminfo", lambda: host)
    monkeypatch.setattr(pf, "read_cgroup_available_gb", lambda *_a: 3.9)  # the compose app's own cap
    real_wait = pf.wait_for_memory

    def wait(est: float, **kw: object) -> bool:  # the real queue, reading the faked host, never sleeping long
        return real_wait(est, read=lambda: host, gpus=lambda: [], poll_s=0.01, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(pf, "wait_for_memory", wait)
    root = _probe_root(tmp_path, "bao-security APPROVED 2026-10-13")
    monkeypatch.setattr(jobrun, "image_digest", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr(jobrun, "git_state", lambda _root: ("c0ffee", False))
    monkeypatch.setattr(jobrun, "read_meminfo", lambda: host)
    guard = SimpleNamespace(should_stop=threading.Event(), min_available_gb=35.0, samples=3, fast_fall_killed=False,
                            thresholds=SimpleNamespace(abort_gb=8.0))  # fmt: skip
    monkeypatch.setattr(jobrun, "run_guarded", lambda launch, *a, **k: (launch(), guard))

    class Launcher:
        def run(self, *a: object, **k: object) -> int:
            return 0

        def state(self, name: str) -> ExitInfo:
            return ExitInfo(exit_code=0, oom_killed=False)

        def stop(self, *a: object) -> None: ...

        def kill(self, *a: object) -> None: ...

        def remove(self, *a: object) -> None: ...

    run_dir = root / "outputs" / "runs" / "20261014-0132-dl-x"
    final = jobrun.run_job(
        ContainerJob(name="20261014-0132-dl-x", image="vlm-x:1", argv=("true",), gpu=False),
        RunManifest(run_id="20261014-0132-dl-x", name="t", stage="data", task="download"),
        run_dir, project_root=root, who="t", est_peak_gb=2.0, lock_path=tmp_path / "GPU_LOCK",
        launcher=Launcher(), memory_wait_s=0.5,  # type: ignore[arg-type]
    )  # fmt: skip
    assert final.status == RunStatus.COMPLETED  # before the fix: SYS-002 after the memory wait


def test_only_the_download_job_is_online(tmp_path: Path) -> None:
    """PM/tester 2026-10-14: the download job inherited HF_HUB_OFFLINE=1 (OfflineModeIsEnabled). It alone is online,
    still only through the egress proxy; every other job stays offline."""
    root, run, _repo, data = _narrow_project(tmp_path)
    auth = "t:" + "ab" * 16
    download = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), egress=True,
                            egress_auth=auth)  # fmt: skip
    online = docker_args(download, root)
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert f"{var}=0" in online
        assert f"{var}=1" not in online
    other = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(data,))
    offline = docker_args(other, root)
    assert "HF_HUB_OFFLINE=1" in offline
    assert "HF_HUB_OFFLINE=0" not in offline
    sneaky = ContainerJob(name="t", image="i", argv=("true",), narrow=True, rw_dirs=(run,),
                          env={"HF_HUB_OFFLINE": "0"})  # fmt: skip
    with pytest.raises(ValueError, match="offline job"):  # security: job.env never turns a normal job online
        docker_args(sneaky, root)


def _race_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[dict[str, object]]) -> object:
    """run_job with the REAL RunnerLauncher polling loop; the runner's replies are scripted."""
    import threading
    from types import SimpleNamespace

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.launcher import RunnerLauncher
    from vlm_pipeline.common.runinfo import RunManifest

    root = _probe_root(tmp_path, "bao-security APPROVED 2026-10-13")
    monkeypatch.setattr(jobrun, "image_digest", lambda _image: f"sha256:{'a' * 64}")
    monkeypatch.setattr(jobrun, "_wait_for_memory", lambda *a, **k: None)
    monkeypatch.setattr(jobrun, "git_state", lambda _root: ("c0ffee", False))
    monkeypatch.setattr(jobrun, "read_meminfo", lambda: SimpleNamespace(mem_available_gb=40.0))
    guard = SimpleNamespace(should_stop=threading.Event(), min_available_gb=35.0, samples=3, fast_fall_killed=False,
                            thresholds=SimpleNamespace(abort_gb=8.0))  # fmt: skip
    monkeypatch.setattr(jobrun, "run_guarded", lambda launch, *a, **k: (launch(), guard))
    monkeypatch.setattr(RunnerLauncher, "POLL_S", 0.0)
    launcher = RunnerLauncher(tmp_path / "runner.sock")
    calls = iter(replies)
    monkeypatch.setattr(launcher, "_call", lambda body: next(calls))
    run_dir = root / "outputs" / "runs" / "20261014-0201-dl-x"
    job = ContainerJob(name="20261014-0201-dl-x", image="vlm-x:1", gpu=False, narrow=True, rw_dirs=(run_dir,),
                       argv=("python", "-m", "vlm_pipeline.download.hub", "--run-dir", "/workspace/x"))  # fmt: skip
    return jobrun.run_job(job, RunManifest(run_id="20261014-0201-dl-x", name="t", stage="data", task="download"),
                          run_dir, project_root=root, who="t", est_peak_gb=2.0, take_gpu_lock=False,
                          launcher=launcher)  # fmt: skip


def test_the_runner_start_race_ends_with_the_jobs_real_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM 2026-10-14: first poll not_found, then running, then exit 1 -> FAILED with exit 1 (download: NET-001),
    never a RunnerError (SYS-000) hiding it."""
    from vlm_pipeline.common.runinfo import RunStatus

    final = _race_run(tmp_path, monkeypatch, [
        {"ok": True, "image_id": "sha256:x"},  # submit
        {"ok": False, "error": "not_found"},  # docker has not created it yet
        {"ok": True, "running": True},
        {"ok": True, "running": False, "exit_code": 1},
        {"ok": True, "running": False, "exit_code": 1},  # state() after the run
        {"ok": True},  # remove
    ])  # fmt: skip
    assert (final.status, final.exit_code) == (RunStatus.FAILED, 1)  # type: ignore[attr-defined]


def test_a_container_that_never_appears_is_a_system_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common.launcher import RunnerError, RunnerLauncher

    monkeypatch.setattr(RunnerLauncher, "START_GRACE_S", 0.05)
    replies = [{"ok": True, "image_id": "sha256:x"}, *({"ok": False, "error": "not_found"} for _ in range(10_000))]
    with pytest.raises(RunnerError, match="not_found"):  # the download maps this to SYS-000, and install.sh fails
        _race_run(tmp_path, monkeypatch, replies)


NEMO_RL_ID = "sha256:4c740d252ef90651621224005623546b4931582fa2eeb9ce816ba9d7576a4add"  # IMAGES.lock, PROBE ONLY


@pytest.mark.parametrize(
    ("image", "groups", "ok"),
    [
        (NEMO_RL_ID, (0,), True),  # its /opt/nemo-rl is root-group only (security 2026-10-14)
        (NEMO_RL_ID, (0, 4), False),  # only the allowed group
        ("vlm-nemo-rl:0.7.0-1", (0,), False),  # a tag is never trusted (anyone can retag another image)
        ("sha256:" + "b" * 64, (0,), False),  # another image, whatever its tag says
        ("vlm-automodel:26.08-6", (), True),  # no extra group: always fine
    ],
)
def test_extra_groups_are_allowed_per_image_only(tmp_path: Path, image: str, groups: tuple[int, ...], ok: bool) -> None:
    root, run, _repo, data = _narrow_project(tmp_path)
    job = ContainerJob(name="t", image=image, argv=("true",), narrow=True, rw_dirs=(run,), ro_inputs=(data,),
                       group_add=groups)  # fmt: skip
    if ok:
        argv = docker_args(job, root)
        assert argv.count("--group-add") == len(groups)
    else:
        with pytest.raises(ValueError, match="may not join group"):
            docker_args(job, root)


@pytest.mark.parametrize(
    ("gpu", "memory_gb", "exempt"), [(False, 4, True), (False, 2, True), (False, 5, False), (True, 4, False)]
)
@pytest.mark.parametrize("narrow", [False, True])
def test_swap_exempt_jobs_get_no_swap_and_a_pids_limit(tmp_path: Path, gpu: bool, memory_gb: int, exempt: bool,
                                                      narrow: bool) -> None:  # fmt: skip
    """ADR-005 amendment 2026-10-14: the free-swap floor is skipped ONLY for a job that cannot use swap at all."""
    from vlm_pipeline.common.jobrun import JOB_PIDS_LIMIT, swap_exempt

    root = Path("/repo")
    if narrow:
        root, run, _repo, _data = _narrow_project(tmp_path)
        job = ContainerJob(name="t", image="img", argv=("true",), gpu=gpu, memory_gb=memory_gb, narrow=True,
                           rw_dirs=(run,))  # fmt: skip
    else:
        job = _job(gpu=gpu, memory_gb=memory_gb)
    args = docker_args(job, root)
    assert swap_exempt(job) is exempt
    pairs = list(itertools.pairwise(args))
    assert ("--memory", f"{memory_gb}g") in pairs
    assert (("--memory-swap", f"{memory_gb}g") in pairs) is exempt  # the flag that makes the exemption safe
    assert (("--pids-limit", str(JOB_PIDS_LIMIT)) in pairs) is exempt
    assert args.count("--memory-swap") == int(exempt)


def test_only_the_swap_floor_is_dropped_for_an_exempt_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from vlm_pipeline.common import jobrun
    from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS

    root, job, manifest = _approved_run(tmp_path, monkeypatch)
    seen: dict[bool, object] = {}
    for exempt_job in (False, True):
        cpu = replace(job, gpu=False, memory_gb=4) if exempt_job else job

        def fake_wait(est: float, *, thresholds: object, _k: bool = exempt_job, **kw: object) -> bool:
            seen[_k] = thresholds
            return False  # cancelled: nothing starts

        monkeypatch.setattr("vlm_pipeline.common.preflight.wait_for_memory", fake_wait)
        jobrun.run_job(cpu, manifest, tmp_path / f"run{exempt_job}", project_root=root, who="t", est_peak_gb=2)  # type: ignore[arg-type]
    assert seen[False] == DEFAULT_THRESHOLDS
    assert seen[True] == replace(DEFAULT_THRESHOLDS, min_swap_free_gb=0.0)
