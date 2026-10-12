"""Host-side job runner: one heavy job = one container (Proposal 008 §2.2, P009 §5, P003 D2/D16).

Order: preflight -> GPU lock -> run.json(running) -> ``docker run`` (no ``--rm``) -> ``docker inspect`` exit code /
OOMKilled -> classify -> run.json(final) -> ``docker rm`` -> release lock. Docker args are built by a pure function
from a fixed permission table, never from user/plugin input. Stdlib + pydantic only (host-safe).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.errors import ExitCode, VlmPipelineError
from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS, MemGuard, MemThresholds, read_meminfo
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
    # Code checkout (git worktree) overlaid read-only on these project dirs, so caches/outputs stay in the main
    # project while the code that runs is exactly the pinned commit (ADR-002 A1).
    code_root: Path | None = None
    code_dirs: tuple[str, ...] = ("src", "configs", "scripts", "docker")
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
    # No phoning home (P003, bao-security): albumentations update check, HF telemetry, generic opt-outs, W&B offline.
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_MODE": "offline",
}
_FORBIDDEN_ENV_PREFIXES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_", "WANDB_API_KEY")
# Anything that looks like a credential (bao-security): OPENAI_API_KEY, AWS_SECRET_ACCESS_KEY, HUGGINGFACE_TOKEN...
# Matched per "_"-separated segment so TOKENIZERS_PARALLELISM & co. pass (fail-closed: KEY_FRAMES needs the allowlist).
_SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_|$)", re.IGNORECASE)
# Explicit exceptions only (none yet; a first-party download job receiving a scoped token would be listed here, D12).
SECRET_ENV_ALLOWLIST: frozenset[str] = frozenset()


def docker_args(job: ContainerJob, project_root: Path, *, lock_token: str | None = None) -> list[str]:
    """Pure: the full ``docker run`` argv for ``job`` (tested; no shell, no user-controlled flags)."""
    env = {**BASE_ENV, **job.env}
    for key in env:
        if key in SECRET_ENV_ALLOWLIST:
            continue
        if key.startswith(_FORBIDDEN_ENV_PREFIXES) or _SECRET_NAME.search(key):
            raise ValueError(f"secret-like env var {key} must not be passed to job containers")
    if lock_token is not None:
        env[gpu_lock.TOKEN_ENV] = lock_token
    mount = f"{project_root}:{CONTAINER_ROOT}" + (":ro" if job.read_only_project else "")
    overlays: list[str] = []
    if job.code_root is not None:
        for d in job.code_dirs:
            overlays += ["-v", f"{job.code_root / d}:{CONTAINER_ROOT}/{d}:ro"]
    args = [
        "docker", "run", "--name", f"vlm-{job.name}",
        "--label", "project=vlm-training",
        "--network", "none",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
        "--user", f"{job.uid}:{job.gid}",
        "--memory", f"{job.memory_gb}g", "--shm-size", f"{job.shm_gb}g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", mount, *overlays, "-w", job.workdir,
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
    preempted_by_memguard: bool = False  # host watchdog stopped the job (MemAvailable < abort_gb)


def classify_exit(info: ExitInfo) -> tuple[RunStatus, str | None]:
    """Status + catalog code (P009 §5, fixcode order). Cancel first; OOM only with evidence; never guess."""
    if info.cancelled_by_executor and info.exit_code in {0, 137, 143, ExitCode.MEM_PREEMPT}:
        return RunStatus.CANCELLED, None
    if info.preempted_by_memguard and info.exit_code != 0:
        return RunStatus.INTERRUPTED, "MEM-002"
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


# Host-side memory watchdog for container jobs (P-1 incident 2026-10-12: memguard was never wired; the peak came
# before step 1, ~1.5 min after start). CUDA on UMA is outside the container cgroup -> host MemAvailable only.
WATCHDOG_INTERVAL_S = 0.5
STOP_GRACE_S = 60


class ThresholdTooLooseError(VlmPipelineError):
    """A job asked for memory thresholds looser than ADR-005 (callers may only tighten them)."""

    code: ClassVar[str] = "CFG-005"


def tightened_thresholds(abort_gb: float | None = None) -> MemThresholds:
    """ADR-005 thresholds, optionally with a HIGHER abort level (e.g. 10 GB for a first measurement run)."""
    if abort_gb is None:
        return DEFAULT_THRESHOLDS
    if not math.isfinite(abort_gb) or abort_gb < DEFAULT_THRESHOLDS.abort_gb:
        raise ThresholdTooLooseError(
            "abort threshold below ADR-005",
            params={"param": "abort_gb", "rule": f">={DEFAULT_THRESHOLDS.abort_gb}", "value": abort_gb},
        )
    warn = max(DEFAULT_THRESHOLDS.warn_gb, abort_gb + 4.0)  # keep warn > abort for the fast-fall rule
    return dataclasses.replace(DEFAULT_THRESHOLDS, abort_gb=abort_gb, warn_gb=warn)


class MemoryPreemptedBeforeStartError(RuntimeError):
    """MemAvailable was already below the abort threshold when the job was about to start (not launched)."""


def run_guarded(
    argv: Sequence[str],
    log_path: Path,
    *,
    on_abort: Callable[[], None],
    hard_exit: Callable[[], None],
    meminfo_path: Path = Path("/proc/meminfo"),
    interval_s: float = WATCHDOG_INTERVAL_S,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
) -> tuple[int, MemGuard]:
    """Run ``argv`` (logged) under a MemGuard that polls BEFORE the process starts and every ``interval_s``.

    Below ``abort_gb``: ``on_abort`` (graceful stop); after the grace or below ``emergency_gb``: ``hard_exit``.
    Returns the exit code and the guard (``guard.should_stop`` tells whether the watchdog stopped the job).
    """
    guard = MemGuard(
        thresholds=thresholds,
        interval_s=interval_s,
        meminfo_path=meminfo_path,
        cgroup_root=None,
        on_abort=on_abort,
        hard_exit=hard_exit,
    )
    guard.start()  # first poll happens here, before anything is launched
    try:
        if guard.should_stop.is_set():
            raise MemoryPreemptedBeforeStartError(f"MemAvailable {guard.min_available_gb:.1f} GB below abort")
        return _run_logged(argv, log_path), guard
    finally:
        guard.stop()


def _docker_stop_async(name: str, grace_s: int = STOP_GRACE_S) -> None:
    # Non-blocking: the watchdog thread must keep polling (emergency kill) while docker waits for the grace period.
    subprocess.Popen(  # noqa: S603
        ["docker", "stop", "-t", str(grace_s), name],  # noqa: S607
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip


def _docker_kill(name: str) -> None:
    subprocess.run(["docker", "kill", name], capture_output=True, check=False)  # noqa: S603,S607


_TRAINABLE_RE = re.compile(r"Trainable parameters percentage:\s*([0-9.]+)\s*%")


def _trainable_pct(log_path: Path) -> float | None:
    """Share of trainable parameters printed by AutoModel (shown in the UI; guards against wrong freezing)."""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _TRAINABLE_RE.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def _docker_state(name: str) -> ExitInfo:
    out = subprocess.run(  # noqa: S603
        ["docker", "inspect", "-f", "{{json .State}}", name],  # noqa: S607 - docker from PATH
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    state = json.loads(out)
    return ExitInfo(exit_code=int(state.get("ExitCode", -1)), oom_killed=bool(state.get("OOMKilled", False)))


_CODE_DIRS = ("src", "configs", "docker", "scripts")


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
        # Untracked code/config is imported or read by runs but absent from HEAD -> also dirty (multi-agent tree).
        untracked = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", *_CODE_DIRS],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip() or untracked.strip())


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
    gate: bool = False,
    cancel: threading.Event | None = None,
    mem_thresholds: MemThresholds = DEFAULT_THRESHOLDS,
) -> RunManifest:
    """Run ``job`` end to end under the GPU lock; always writes a final run.json and releases the lock."""
    from vlm_pipeline.common.preflight import preflight

    if gate:  # gate/published runs must be reproducible before they take the GPU (ADR-002 A1)
        from vlm_pipeline.common.runinfo import reproducibility_gaps

        commit, dirty = git_state(job.code_root or project_root)
        probe = manifest.model_copy(
            update={
                "git_commit": manifest.git_commit or commit,
                "git_dirty": dirty,
                "container_digest": manifest.container_digest or image_digest(job.image),
            }
        )
        gaps = reproducibility_gaps(probe)
        if gaps:
            raise RuntimeError(f"gate run refused, not reproducible: missing/invalid {gaps}")
    verdict = preflight(est_peak_gb, ckpt_gb=ckpt_gb)
    if not verdict.ok:
        raise RuntimeError(f"preflight refused: {verdict.to_json()}")
    path = lock_path or gpu_lock.default_lock_path()
    handle = gpu_lock.acquire(path, who=who, job=manifest.name, run_id=manifest.run_id, host_pid=os.getpid())
    container = f"vlm-{job.name}"
    try:
        commit, dirty = git_state(job.code_root or project_root)
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
        baseline_gb = read_meminfo().mem_available_gb
        cancelled = threading.Event()
        done = threading.Event()

        def _watch_cancel() -> None:  # executor cancel -> graceful docker stop (job can save a checkpoint)
            assert cancel is not None  # noqa: S101 - only started when cancel is given
            while not done.is_set():
                if cancel.wait(0.5):
                    cancelled.set()
                    _docker_stop_async(container)
                    return

        if cancel is not None:
            threading.Thread(target=_watch_cancel, name=f"cancel-{container}", daemon=True).start()
        _, guard = run_guarded(
            docker_args(job, project_root, lock_token=handle.info.token),
            run_dir / "logs" / "stdout.log",
            on_abort=lambda: _docker_stop_async(container),
            hard_exit=lambda: _docker_kill(container),
            thresholds=mem_thresholds,
        )
        done.set()
        state = _docker_state(container)
        info = dataclasses.replace(
            state, preempted_by_memguard=guard.should_stop.is_set(), cancelled_by_executor=cancelled.is_set()
        )
        status, code = classify_exit(info)
        manifest = manifest.model_copy(
            update={
                "status": status,
                "error_code": code,
                "exit_code": info.exit_code,
                "oom_killed": info.oom_killed,
                "ended_at": time.time(),
                # Host-level estimate (includes CUDA on UMA, which the container cgroup misses) + shared-host noise.
                "peak_mem_gb": round(max(0.0, baseline_gb - guard.min_available_gb), 1),
                "host_min_available_gb": round(guard.min_available_gb, 1),
                "trainable_pct": _trainable_pct(run_dir / "logs" / "stdout.log"),
                "memguard_fast_fall": bool(guard.fast_fall_killed),
                "memguard_abort_gb": mem_thresholds.abort_gb,
            }
        )
        write_manifest(run_dir, manifest)
        return manifest
    except BaseException as exc:
        if not manifest.status.is_terminal:  # runner itself failed: never leave "running" behind
            # Not launched because memory was already below abort -> MEM-001 (insufficient), not an unknown error.
            code = "MEM-001" if isinstance(exc, MemoryPreemptedBeforeStartError) else "SYS-000"
            write_manifest(
                run_dir,
                manifest.model_copy(update={"status": RunStatus.FAILED, "error_code": code, "ended_at": time.time()}),
            )
        raise
    finally:
        _docker_rm(container)
        gpu_lock.release(handle)
