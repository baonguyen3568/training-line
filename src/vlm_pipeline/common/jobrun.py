"""Host-side job runner: one heavy job = one container (Proposal 008 §2.2, P009 §5, P003 D2/D16).

Order: preflight -> GPU lock -> run.json(running) -> ``docker run`` (no ``--rm``) -> ``docker inspect`` exit code /
OOMKilled -> classify -> run.json(final) -> ``docker rm`` -> release lock. Docker args are built by a pure function
from a fixed permission table, never from user/plugin input. Stdlib + pydantic only (host-safe).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.errors import ExitCode
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest

CONTAINER_ROOT = "/workspace"
CANCEL_GRACE_S = 120


@dataclass(frozen=True)
class ContainerJob:
    """What to run. ``argv`` runs inside ``image`` with the project mounted at ``/workspace``."""

    name: str  # container name suffix: vlm-<name>
    image: str
    argv: tuple[str, ...]
    gpu: bool = True
    memory_gb: int = 48  # caps CPU RSS only; CUDA on UMA is guarded by preflight + memguard
    shm_gb: int = 8
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str = CONTAINER_ROOT
    read_only_project: bool = False
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)


# Env every first-party job gets (P003 D2, ADR-005 D1 caches); never secrets.
BASE_ENV: Mapping[str, str] = {
    "HOME": "/tmp",  # noqa: S108 - HOME inside the container (non-root uid has no passwd home)
    "HF_HOME": f"{CONTAINER_ROOT}/.cache/hf",
    "HF_DATASETS_CACHE": f"{CONTAINER_ROOT}/.cache/hf/datasets",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TRITON_CACHE_DIR": f"{CONTAINER_ROOT}/.cache/triton",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": f"{CONTAINER_ROOT}/src",
    "VLM_PROJECT_ROOT": CONTAINER_ROOT,
}
_FORBIDDEN_ENV_PREFIXES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_", "WANDB_API_KEY")


def docker_args(job: ContainerJob, project_root: Path, *, lock_token: str | None = None) -> list[str]:
    """Pure: the full ``docker run`` argv for ``job`` (tested; no shell, no user-controlled flags)."""
    env = {**BASE_ENV, **job.env}
    for key in env:
        if key.startswith(_FORBIDDEN_ENV_PREFIXES):
            raise ValueError(f"secret-like env var {key} must not be passed to job containers")
    if lock_token is not None:
        env[gpu_lock.TOKEN_ENV] = lock_token
    mount = f"{project_root}:{CONTAINER_ROOT}" + (":ro" if job.read_only_project else "")
    args = [
        "docker", "run", "--name", f"vlm-{job.name}",
        "--label", "project=vlm-training",
        "--network", "none",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
        "--user", f"{job.uid}:{job.gid}",
        "--memory", f"{job.memory_gb}g", "--shm-size", f"{job.shm_gb}g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", mount, "-w", job.workdir,
    ]  # fmt: skip
    if job.gpu:
        args += ["--gpus", "all"]
    for key, value in sorted(env.items()):
        args += ["-e", f"{key}={value}"]
    return [*args, job.image, *job.argv]


@dataclass(frozen=True)
class ExitInfo:
    exit_code: int
    oom_killed: bool
    cancelled_by_executor: bool = False


def classify_exit(info: ExitInfo) -> tuple[RunStatus, str | None]:
    """Status + catalog code (P009 §5, fixcode order). Cancel first; OOM only with evidence; never guess."""
    if info.cancelled_by_executor and info.exit_code in {0, 137, 143, ExitCode.MEM_PREEMPT}:
        return RunStatus.CANCELLED, None
    if info.exit_code == 0:
        return RunStatus.COMPLETED, None
    if info.exit_code == ExitCode.MEM_PREEMPT:
        return RunStatus.INTERRUPTED, "MEM-002"
    if info.oom_killed:
        return RunStatus.FAILED, "MEM-003"
    return RunStatus.FAILED, "SYS-003"


Runner = Callable[[Sequence[str], Path], int]


def _run_logged(argv: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 - argv built by docker_args
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            return proc.wait()


def _docker_state(name: str) -> ExitInfo:
    out = subprocess.run(  # noqa: S603
        ["docker", "inspect", "-f", "{{json .State}}", name],  # noqa: S607 - docker from PATH
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    state = json.loads(out)
    return ExitInfo(exit_code=int(state.get("ExitCode", -1)), oom_killed=bool(state.get("OOMKilled", False)))


def git_state(root: Path) -> tuple[str | None, bool | None]:
    """(HEAD commit, dirty) of the project; (None, None) if git is unavailable."""
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip())


def image_digest(image: str) -> str | None:
    """Local image content id (``sha256:...``): identifies exactly what ran, also for local builds."""
    try:
        return subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", "-f", "{{.Id}}", image],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def _docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", name], capture_output=True, check=False)  # noqa: S603,S607


def run_job(
    job: ContainerJob,
    manifest: RunManifest,
    run_dir: Path,
    *,
    project_root: Path,
    who: str,
    est_peak_gb: float,
    ckpt_gb: float = 0.0,
    lock_path: Path | None = None,
) -> RunManifest:
    """Run ``job`` end to end under the GPU lock; always writes a final run.json and releases the lock."""
    from vlm_pipeline.common.preflight import preflight

    verdict = preflight(est_peak_gb, ckpt_gb=ckpt_gb)
    if not verdict.ok:
        raise RuntimeError(f"preflight refused: {verdict.to_json()}")
    path = lock_path or gpu_lock.default_lock_path()
    handle = gpu_lock.acquire(path, who=who, job=manifest.name, run_id=manifest.run_id, host_pid=os.getpid())
    container = f"vlm-{job.name}"
    try:
        commit, dirty = git_state(project_root)
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "pid": os.getpid(),
                "git_commit": manifest.git_commit or commit,
                "git_dirty": dirty if manifest.git_dirty is None else manifest.git_dirty,
                "container_image": manifest.container_image or job.image,
                "container_digest": manifest.container_digest or image_digest(job.image),
            }
        )
        write_manifest(run_dir, manifest)
        _run_logged(docker_args(job, project_root, lock_token=handle.info.token), run_dir / "logs" / "stdout.log")
        info = _docker_state(container)
        status, code = classify_exit(info)
        manifest = manifest.model_copy(
            update={
                "status": status,
                "error_code": code,
                "exit_code": info.exit_code,
                "oom_killed": info.oom_killed,
                "ended_at": time.time(),
            }
        )
        write_manifest(run_dir, manifest)
        return manifest
    except BaseException:
        if not manifest.status.is_terminal:  # runner itself failed: never leave "running" behind
            write_manifest(
                run_dir,
                manifest.model_copy(
                    update={"status": RunStatus.FAILED, "error_code": "SYS-000", "ended_at": time.time()}
                ),
            )
        raise
    finally:
        _docker_rm(container)
        gpu_lock.release(handle)
