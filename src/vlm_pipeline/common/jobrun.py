"""Host-side job runner: one heavy job = one container (Proposal 008 §2.2, P009 §5, P003 D2/D16).

Order: preflight -> GPU lock -> run.json(running) -> ``docker run`` (no ``--rm``) -> ``docker inspect`` exit code /
OOMKilled -> classify -> run.json(final) -> ``docker rm`` -> release lock. Docker args are built by a pure function
from a fixed permission table, never from user/plugin input. Stdlib + pydantic only (host-safe).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from vlm_pipeline.common import gpu_lock
from vlm_pipeline.common.errors import ExitCode, VlmPipelineError
from vlm_pipeline.common.host_paths import check_mount_source, link_units, secret_file_problem, tree_problem
from vlm_pipeline.common.images import check_pinned_image
from vlm_pipeline.common.launcher import JobLauncher, default_launcher
from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS, MemGuard, MemThresholds, read_meminfo
from vlm_pipeline.common.paths import CODE_ROOT_ENV, is_team_worktree_dir
from vlm_pipeline.common.run_claims import RunDirClaimedError, claim_run, is_claimed
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
    # With ``read_only_project``: the only writable host dirs (each under <project>/outputs/), mounted at their own
    # container path. Jobs that run untrusted input (trial inference on user images, remote model code) write their
    # result here and nowhere else (bao-security, B-7).
    rw_dirs: tuple[Path, ...] = ()
    # Replaces the image's ENTRYPOINT (e.g. vllm/vllm-openai's ["vllm", "serve"]) so ``argv`` is the whole command.
    entrypoint: str | None = None
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)
    # P017 step 6 (narrowed mounts, bao-security): no project mount at all. The job sees src/ + configs/ (ro), the
    # single writable dir ``rw_dirs[0]``, each of ``ro_inputs`` (host paths under the project/data root, ro, at their
    # own container path) and ``code_files`` (single files, e.g. a worker script, ro). Compile caches live in the
    # writable dir (never shared between jobs). App workers set it; team CLI jobs keep the project mount for now.
    narrow: bool = False
    ro_inputs: tuple[Path, ...] = ()
    code_files: tuple[str, ...] = ()
    # Download jobs only (P018, security download-egress §1-2), set by vlm-runner from the job kind, never a request
    # field: the internal egress network (whose only way out is the HF-only proxy) and the token as a mounted FILE.
    egress: bool = False
    secret_file: Path | None = None
    egress_auth: str | None = None  # "<job>:<key>" registered with the proxy (its byte budget); not the HF token
    # What the APP hands vlm-runner for a download job (RunnerLauncher sends both; never on a docker argv): the
    # proxy's byte budget and the HF token, kept in memory only (repr=False: never in a log or a traceback).
    download_budget: int | None = None
    hf_token: str | None = field(default=None, repr=False)


# Env every first-party job gets (P003 D2, ADR-005 D1 caches); never secrets.
logger = logging.getLogger(__name__)

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


_ENTRYPOINT = re.compile(r"[A-Za-z0-9_./-]{1,64}")


def _entrypoint_args(entrypoint: str | None) -> list[str]:
    """One plain token (bao-security): no spaces, never starting with "-", so it can't become a docker flag."""
    if entrypoint is None:
        return []
    if not _ENTRYPOINT.fullmatch(entrypoint) or entrypoint.startswith("-"):
        raise ValueError(f"invalid entrypoint {entrypoint!r}")
    return ["--entrypoint", entrypoint]


def validate_rw_dirs(project_root: Path, dirs: Iterable[Path]) -> list[Path]:
    """Resolved writable dirs, each strictly under <project>/outputs/ and not in outputs/system (host logs, locks).

    Call it BEFORE creating any of them: a rejected dir must never be created (bao-security).
    """
    outputs = (project_root / "outputs").resolve()
    system = outputs / "system"
    resolved = []
    for rw in dirs:
        host = rw.resolve()
        if outputs not in host.parents or host == system or system in host.parents:
            raise ValueError(f"rw dir {rw} must be inside {outputs} (not outputs/system: host logs, locks)")
        resolved.append(host)
    return resolved


_SCRATCH = "/tmp/.cache"  # noqa: S108 - per-container scratch (dies with the container)


def compile_cache_env(job: ContainerJob, project_root: Path, image_id: str | None) -> dict[str, str]:
    """JIT/inductor caches that outlive the container, keyed by the image id (a new image never loads an old .so).

    research: modelopt compiled its CUDA extension (~3 min, many cc1plus) in EVERY job because HOME=/tmp. A
    read-only project keeps them in the job's own writable dir. ``MAX_JOBS=2`` caps parallel compiler processes.
    """
    if not image_id:  # no image identity: never persist a .so another image could load (bao-fixcode)
        scratch = {"TORCH_EXTENSIONS_DIR": "torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
        if job.read_only_project:
            scratch["TRITON_CACHE_DIR"] = "triton"
        return {**{k: f"{_SCRATCH}/{v}" for k, v in scratch.items()}, "MAX_JOBS": "2"}
    key = image_id.removeprefix("sha256:")[:16]
    base = f"{CONTAINER_ROOT}/.cache"
    # A read-only job (trial runs remote code on a user image) caches in ITS OWN run dir, never in a shared writable
    # cache: a shared one would let it plant a .so for the next job. Cost: no speed-up across runs for those jobs.
    if job.read_only_project and job.rw_dirs:
        rel = job.rw_dirs[0].resolve().relative_to(project_root.resolve()).as_posix()
        base = f"{CONTAINER_ROOT}/{rel}/.cache"
    elif job.read_only_project:
        base = _SCRATCH  # nothing writable in the project: rebuilt per job, as before
    env = {
        "TORCH_EXTENSIONS_DIR": f"{base}/torch_extensions/{key}",
        "TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor/{key}",
        "MAX_JOBS": "2",
    }
    if job.read_only_project:  # BASE_ENV's Triton cache is in the (read-only) project: Triton autotune fails there
        env["TRITON_CACHE_DIR"] = f"{base}/triton/{key}"
    return env


TMPFS_GB = 8  # /tmp in a narrowed job: RAM on UMA, so bounded; exec allowed (Triton/torch build fallbacks run there)
NARROW_CODE_DIRS = ("src", "configs")


def _narrow_cache_env(rw_rel: str) -> dict[str, str]:
    """Every cache a job writes goes to its own writable dir: nothing is shared with another job (step 6 Q2)."""
    base = f"{CONTAINER_ROOT}/{rw_rel}/.cache"
    return {
        "TORCH_EXTENSIONS_DIR": f"{base}/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor",
        "TRITON_CACHE_DIR": f"{base}/triton",
        "HF_DATASETS_CACHE": f"{base}/hf_datasets",
        "HF_MODULES_CACHE": f"{base}/hf_modules",
        "MAX_JOBS": "2",
    }


_HUB_REPO = re.compile(r"^(?:models|datasets)--[A-Za-z0-9][A-Za-z0-9._-]*--[A-Za-z0-9][A-Za-z0-9._-]*$")


def _input_problem(rel: Path) -> str | None:
    """None when ``rel`` is under an ALLOWED input root (bao-security S6-1: an allowlist, never "anything but").

    One HF hub repo (or deeper), one prepared model, one run (graph node or CLI), one folder under data/, one upload.
    Never a root itself (``.cache``, ``.cache/hf``, ``outputs/runs``, ``data``): those hold other users' data, caches
    and the HF login token.
    """
    p = rel.parts
    if not p or ".." in p or any(part.startswith(".env") for part in p):
        return "outside the project or a secrets file"
    allowed = (
        (p[:3] == (".cache", "hf", "hub") and len(p) >= 4 and _HUB_REPO.fullmatch(p[3]) is not None)
        or (p[:2] == (".cache", "prepared") and len(p) >= 3)
        or (p[:3] == ("outputs", "graph", "runs") and len(p) >= 4)
        or (p[:2] == ("outputs", "runs") and len(p) >= 3)
        or (p[:1] == ("data",) and len(p) >= 2)
        or (p[:1] == ("uploads",) and len(p) == 2)
    )
    if allowed and p[0] == "outputs" and ".cache" in p:  # another job's own .so / triton cache: never handed on
        return "a job cache folder"
    return None if allowed else "not under an allowed input root"


def _checked_input(root: Path, rw: Path | None, given: Path) -> tuple[Path, set[Path]]:
    """(path relative to the root, extra link units) of one read-only input, or ValueError naming the rule."""
    if not given.is_relative_to(root):
        raise ValueError(f"input {given} is outside the project")
    rel = given.relative_to(root)
    if (problem := _input_problem(rel)) is not None:
        raise ValueError(f"input {rel}: {problem}")
    if rw is not None and (given == rw or given.is_relative_to(rw) or rw.is_relative_to(given)):
        raise ValueError(f"input {rel} overlaps the writable dir")
    is_dir = given.is_dir() and not given.is_symlink()
    check_mount_source(root, given, directory=is_dir)
    if not is_dir:
        return rel, set()
    if (secret := secret_file_problem(given)) is not None:
        raise ValueError(f"input {rel}: {secret} looks like a credential")
    if rel.parts[:1] == (".cache",):  # HF caches link into repos / the blob store: follow, mount those units
        problem, units = link_units(root, given)
        if problem is not None:
            raise ValueError(f"input {rel}: {problem}")
        return rel, units
    if (problem := tree_problem(given)) is not None:  # job-written data: no link at all
        raise ValueError(f"input {rel}: {problem} is not a plain file")
    return rel, set()


class JobInputRefusedError(VlmPipelineError):
    """A read-only input of a narrowed job breaks the mount rules: refused before any run dir is written.

    ``params["path"]`` is the input as the user knows it (relative to the project when inside it); the rule that
    refused it is the message (technical detail only).
    """

    code: ClassVar[str] = "SEC-001"


def input_problem(project_root: Path, path: Path) -> str | None:
    """Why ``path`` cannot be a read-only input of a narrowed job, or None. Same rules as ``narrow_mounts``.

    For callers that build inputs (``run_vllm_batch``...) and want to refuse early; ``run_job`` checks again.
    """
    root = project_root.resolve()
    try:
        _checked_input(root, None, Path(os.path.normpath(path.absolute())))
    except ValueError as exc:
        return str(exc)
    return None


def _refuse_bad_inputs(job: ContainerJob, project_root: Path) -> None:
    root = project_root.resolve()
    rw = validate_rw_dirs(root, job.rw_dirs)[0] if len(job.rw_dirs) == 1 else None
    for given in sorted({Path(os.path.normpath(p.absolute())) for p in job.ro_inputs}):
        try:
            _checked_input(root, rw, given)
        except ValueError as exc:
            shown = given.relative_to(root).as_posix() if given.is_relative_to(root) else given.name
            raise JobInputRefusedError(str(exc), params={"path": shown}) from None


def narrow_mounts(job: ContainerJob, project_root: Path) -> list[str]:
    """``-v``/``--tmpfs`` args of a narrowed job (P017 step 6). Every source is checked on the host first.

    One writable dir (the run dir); src/ + configs/ + ``code_files`` read-only; each input read-only at its own
    container path, never the run dir or a parent/child of it, never under outputs/system, .git, .team or a .env*;
    the walk from the root to each source has no symlink (``check_mount_source``) and the tree itself holds plain
    files only, except an HF hub repo whose RELATIVE links stay inside it (``tree_problem``).
    """
    root = project_root.resolve()
    if len(job.rw_dirs) != 1:
        raise ValueError("a narrowed job has exactly one writable dir")
    rw = validate_rw_dirs(root, job.rw_dirs)[0]
    check_mount_source(root, rw, directory=True)
    rw_rel = rw.relative_to(root)
    code = (job.code_root or root).resolve()
    args = ["--tmpfs", f"/tmp:rw,exec,nosuid,size={TMPFS_GB}g"]  # noqa: S108 - container path
    for d in NARROW_CODE_DIRS:
        args += ["-v", f"{check_mount_source(code, code / d, directory=True)}:{CONTAINER_ROOT}/{d}:ro"]
    for f in job.code_files:
        args += ["-v", f"{check_mount_source(code, code / f, directory=False)}:{CONTAINER_ROOT}/{f}:ro"]
    args += ["-v", f"{rw}:{CONTAINER_ROOT}/{rw_rel.as_posix()}"]
    extra: set[Path] = set()
    mounted: list[Path] = []
    for given in sorted({Path(os.path.normpath(p.absolute())) for p in job.ro_inputs}):
        rel, units = _checked_input(root, rw, given)
        extra |= units
        mounted.append(given)
        args += ["-v", f"{given}:{CONTAINER_ROOT}/{rel.as_posix()}:ro"]
    for unit in sorted(extra):  # what those links pass through, not already inside a mounted input
        if any(unit == m or unit.is_relative_to(m) for m in mounted):
            continue
        check_mount_source(root, unit, directory=unit.is_dir())
        args += ["-v", f"{unit}:{CONTAINER_ROOT}/{unit.relative_to(root).as_posix()}:ro"]
    return args


def docker_args(
    job: ContainerJob, project_root: Path, *, lock_token: str | None = None, image_id: str | None = None
) -> list[str]:
    """Pure: the full ``docker run`` argv for ``job`` (tested; no shell, no user-controlled flags)."""
    if job.narrow:
        return _narrow_docker_args(job, project_root, lock_token=lock_token)
    _no_egress(job)
    env = {**BASE_ENV, **compile_cache_env(job, project_root, image_id), **job.env}  # job.env wins
    for key in env:
        if key in SECRET_ENV_ALLOWLIST:
            continue
        if key.startswith(_FORBIDDEN_ENV_PREFIXES) or _SECRET_NAME.search(key):
            raise ValueError(f"secret-like env var {key} must not be passed to job containers")
    if lock_token is not None:
        env[gpu_lock.TOKEN_ENV] = lock_token
    mount = f"{project_root}:{CONTAINER_ROOT}" + (":ro" if job.read_only_project else "")
    overlays: list[str] = []
    for host in validate_rw_dirs(project_root, job.rw_dirs):
        overlays += ["-v", f"{host}:{CONTAINER_ROOT}/{host.relative_to(project_root.resolve()).as_posix()}"]
    # SEC-005: the project mount must not expose secrets files (.env holds the HF token; jobs run as the same uid).
    # Each root-level .env* is covered by an empty read-only file. Narrower mounts (P015) are the long-term fix.
    for secret in sorted(project_root.glob(".env*")):
        if secret.is_file() or secret.is_symlink():
            overlays += ["-v", f"/dev/null:{CONTAINER_ROOT}/{secret.name}:ro"]
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
    args += _entrypoint_args(job.entrypoint)
    return [*args, job.image, *job.argv]


def _check_env(env: Mapping[str, str]) -> None:
    for key in env:
        if key in SECRET_ENV_ALLOWLIST:
            continue
        if key.startswith(_FORBIDDEN_ENV_PREFIXES) or _SECRET_NAME.search(key):
            raise ValueError(f"secret-like env var {key} must not be passed to job containers")


EGRESS_NETWORK = "ntc-training_egress"  # compose network `egress` (internal: true); only egress-proxy leaves it
EGRESS_PROXY = "http://egress-proxy:3128"
_EGRESS_AUTH = re.compile(r"^[a-z0-9][a-z0-9-]{0,120}:[0-9a-f]{32,128}$")
SECRET_TOKEN_PATH = "/run/secrets/hf_token"  # noqa: S105 - a path: the job reads the token from this file, never env


def _no_egress(job: ContainerJob) -> None:
    if job.egress or job.secret_file is not None:
        raise ValueError("network and token files are for narrowed download jobs only")


def _egress_args(job: ContainerJob) -> tuple[list[str], dict[str, str]]:
    """(docker args, env) of a download job's network and token; ``--network none`` for every other job."""
    if not job.egress:
        if job.secret_file is not None:
            raise ValueError("a token file is only for a download job")
        return ["--network", "none"], {}
    if job.egress_auth is None or not _EGRESS_AUTH.fullmatch(job.egress_auth):
        raise ValueError("a download job needs its proxy credential")
    proxy = EGRESS_PROXY.replace("http://", f"http://{job.egress_auth}@", 1)
    env = {"HTTPS_PROXY": proxy, "NO_PROXY": "", "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    args = ["--network", EGRESS_NETWORK]
    if job.secret_file is not None:
        st = os.lstat(job.secret_file)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("the token file must be a regular file")
        args += ["-v", f"{job.secret_file}:{SECRET_TOKEN_PATH}:ro"]
        env["HF_TOKEN_PATH"] = SECRET_TOKEN_PATH
    return args, env


def _narrow_docker_args(job: ContainerJob, project_root: Path, *, lock_token: str | None) -> list[str]:
    mounts = narrow_mounts(job, project_root)
    rw_rel = job.rw_dirs[0].resolve().relative_to(project_root.resolve()).as_posix()
    network, egress_env = _egress_args(job)
    env = {**BASE_ENV, **_narrow_cache_env(rw_rel), **job.env}  # job.env wins
    _check_env(env)
    env.update(egress_env)  # set here, never from job.env: HF_TOKEN_PATH names the mounted file, not a secret
    if lock_token is not None:
        env[gpu_lock.TOKEN_ENV] = lock_token
    args = [
        "docker", "run", "--name", f"vlm-{job.name}",
        "--label", "project=vlm-training", "--label", f"vlm.job={job.name}",  # the runner acts only on its own jobs
        *network,
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
        "--user", f"{job.uid}:{job.gid}",
        "--memory", f"{job.memory_gb}g", "--shm-size", f"{job.shm_gb}g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        *mounts, "-w", job.workdir,
    ]  # fmt: skip
    if job.gpu:
        args += ["--gpus", "all"]
    for key, value in sorted(env.items()):
        args += ["-e", f"{key}={value}"]
    args += _entrypoint_args(job.entrypoint)
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


MIN_FREE_FRAC = 0.1  # below this no job could ever start; keeps warn_free_frac (ceiling - 0.05) positive


def tightened_thresholds(abort_gb: float | None = None, max_free_frac: float | None = None) -> MemThresholds:
    """ADR-005 thresholds, optionally STRICTER; anything looser is refused (CFG-005).

    ``abort_gb``: a higher abort level (e.g. 10 GB for a first measurement run). ``max_free_frac``: a lower share
    of the free memory a job may use than the user's 80 %.
    """
    thresholds = DEFAULT_THRESHOLDS
    if abort_gb is not None:
        if not math.isfinite(abort_gb) or abort_gb < DEFAULT_THRESHOLDS.abort_gb:
            raise ThresholdTooLooseError(
                "abort threshold below ADR-005",
                params={"param": "abort_gb", "rule": f">={DEFAULT_THRESHOLDS.abort_gb}", "value": abort_gb},
            )
        warn = max(DEFAULT_THRESHOLDS.warn_gb, abort_gb + 4.0)  # keep warn > abort for the fast-fall rule
        thresholds = dataclasses.replace(thresholds, abort_gb=abort_gb, warn_gb=warn)
    if max_free_frac is not None:
        ceiling = DEFAULT_THRESHOLDS.max_free_frac
        if not math.isfinite(max_free_frac) or not MIN_FREE_FRAC <= max_free_frac <= ceiling:
            raise ThresholdTooLooseError(
                "free-memory share outside the allowed range",
                params={"param": "max_free_frac", "rule": f"[{MIN_FREE_FRAC}, {ceiling}]", "value": max_free_frac},
            )
        warn_frac = min(DEFAULT_THRESHOLDS.warn_free_frac, max_free_frac - 0.05)
        thresholds = dataclasses.replace(thresholds, max_free_frac=max_free_frac, warn_free_frac=warn_frac)
    return thresholds


class MemoryPreemptedBeforeStartError(RuntimeError):
    """MemAvailable was already below the abort threshold when the job was about to start (not launched)."""


def run_guarded(
    argv: Sequence[str] | Callable[[], int],
    log_path: Path,
    *,
    on_abort: Callable[[], None],
    hard_exit: Callable[[], None],
    meminfo_path: Path = Path("/proc/meminfo"),
    interval_s: float = WATCHDOG_INTERVAL_S,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    on_guard_ready: Callable[[MemGuard], None] | None = None,
) -> tuple[int, MemGuard]:
    """Run ``argv`` (logged) or a launcher's blocking start, under a MemGuard polling before start and every interval.

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
    if on_guard_ready is not None:  # levels are resolved from A0 now: let the caller publish them while the job runs
        on_guard_ready(guard)
    try:
        if guard.should_stop.is_set():
            raise MemoryPreemptedBeforeStartError(f"MemAvailable {guard.min_available_gb:.1f} GB below abort")
        rc = argv() if callable(argv) else _run_logged(argv, log_path)
        return rc, guard
    finally:
        guard.stop()


_SIGNAL_STOP_RE = re.compile(r"Received signal \d+, initiating graceful stop")


def _stopped_by_signal(log_path: Path) -> bool:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            return any(_SIGNAL_STOP_RE.search(line) for line in fh)
    except OSError:
        return False


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


# Engine log lines that mean "this run is probably not doing what the recipe intends" -> run.json engine_warnings.
# Patterns follow the engine's wording: re-check tests/fixtures/engine_warnings/ whenever the image changes.
# Not scanned when the runner itself fails (engine_warnings stays None = unknown).
ENGINE_WARNINGS: dict[str, re.Pattern[str]] = {
    # P-3: bf16 params + torch AdamW, no fp32 master -> most updates rounded away (001a §10.9).
    "bf16_adam_params": re.compile(r"Detected torch\.optim\.Adam/AdamW with trainable bf16 model parameters"),
}


def _engine_warnings(log_path: Path) -> list[str]:
    """Known-bad engine warnings found in the job log (codes from ``ENGINE_WARNINGS``, in table order)."""
    found: set[str] = set()
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                found.update(code for code, rx in ENGINE_WARNINGS.items() if rx.search(line))
    except OSError:
        return []
    return [code for code in ENGINE_WARNINGS if code in found]


_CODE_DIRS = ("src", "configs", "docker", "scripts")


def git_state(root: Path, dirs: tuple[str, ...] = _CODE_DIRS) -> tuple[str | None, bool | None]:
    """(HEAD commit, dirty under ``dirs``) of the project; (None, None) if git is unavailable."""
    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", *dirs],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        # Untracked code/config is imported or read by runs but absent from HEAD -> also dirty (multi-agent tree).
        untracked = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "--", *dirs],  # noqa: S607
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip() or untracked.strip())


def image_digest(image: str) -> str | None:
    """Local image id of ``image`` (None when it is missing): the host docker CLI, or vlm-runner in compose."""
    return default_launcher().image_id(image)


def _check_gate(job: ContainerJob, manifest: RunManifest, project_root: Path) -> None:
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


class PreflightRefusedError(VlmPipelineError):
    """A preflight condition that waiting does not fix (disk, stale/corrupt lock, no GPU...): its catalog code."""

    def __init__(self, code: str, reason: str, params: Mapping[str, object]) -> None:
        super().__init__(reason, params=params)
        self.code = code  # type: ignore[misc]  # the finding's own catalog code (MEM-001, DISK-001, SYS-001...)


GPU_POLL_S = 10.0


def _queue_info(since: float, *, busy: bool, short: Any) -> dict[str, float]:
    """What a queued job reports on every poll: GPU busy, and/or the numbers of the memory budget that blocks it."""
    info: dict[str, float] = {"since": since, "waited_s": time.time() - since}
    if busy:
        info["gpu_busy"] = 1.0
    if short is not None:  # the UI shows "waiting for free memory" (host) or the VRAM figures (discrete GPU)
        prefix = "vram_" if short.params.get("memory") == "vram" else ""
        for key in ("need_gb", "avail_gb"):
            value = short.params.get(key)
            if isinstance(value, int | float):
                info[prefix + key] = float(value)
    return info


def _memory_timeout(finding: Any, waited_s: float) -> Any:
    """The SYS-002 finding of a memory shortage that outlasted the wait (bao-design text variants memory / vram)."""
    from vlm_pipeline.common.preflight import Finding

    vram = finding.params.get("memory") == "vram"
    params = {
        "minutes": math.ceil(waited_s / 60),
        "waiting_for": "vram" if vram else "memory",
        "need_gb": round(float(finding.params.get("need_gb", 0.0)), 1),
        "avail_gb": round(float(finding.params.get("avail_gb", 0.0)), 1),
    }
    return Finding("SYS-002", ExitCode.NO_MEM, f"still short of memory after the wait: {finding.reason}", params)


def _wait_for_gpu(
    manifest: RunManifest,
    run_dir: Path,
    path: Path,
    est_peak_gb: float,
    ckpt_gb: float,
    *,
    who: str,
    project_root: Path,
    cancel: threading.Event | None,
    on_wait: Callable[[dict[str, float]], None] | None,
    timeout_s: float,
    thresholds: MemThresholds,
    memory_wait_s: float = 1800.0,
) -> gpu_lock.LockHandle | RunManifest:
    """Preflight, then take the lock; QUEUE while another live job holds it (research R2: never SYS-000 for "busy").

    A busy lock with a live holder is waited for (``on_wait({"gpu_busy": 1, ...})``, cancel stops the wait). Not
    enough free memory (MEM-001, host or VRAM) is waited for too: on a shared host the memory free when the job
    queued has often changed by the time the lock frees up (PM 2026-10-13: queue, never fail); ``on_wait`` then
    carries need/avail, and after ``memory_wait_s`` of continuous shortage the job fails with SYS-002
    (waiting_for memory | vram). A stale or unreadable lock and every other refusal fail at once with the finding's
    own catalog code and params (a lock is never removed here). A lost race on ``acquire`` just waits again.
    Returns the lock, or the CANCELLED manifest.
    """
    from vlm_pipeline.common.diskguard import check_lock
    from vlm_pipeline.common.preflight import preflight

    since = time.time()
    short_since: float | None = None  # start of the current memory shortage
    while True:
        verdict = preflight(est_peak_gb, ckpt_gb=ckpt_gb, thresholds=thresholds, lock_holder=lambda: check_lock(path))
        busy = [f for f in verdict.findings if f.code == "SYS-001" and f.params.get("state") == "busy"]
        short = [f for f in verdict.findings if f.code == "MEM-001"]  # transient on a shared host: wait
        refused = next((f for f in verdict.findings if f not in busy and f not in short), None)
        if refused is None and busy and time.time() - since > timeout_s:
            refused = busy[0]
        # Only a shortage while the GPU is FREE counts: behind a busy lock the memory is mostly the holder's (backend)
        short_since = (short_since or time.time()) if short and not busy else None
        if refused is None and short and short_since is not None and time.time() - short_since > memory_wait_s:
            refused = _memory_timeout(short[0], time.time() - short_since)
        if refused is not None:
            failed = {"status": RunStatus.FAILED, "error_code": refused.code, "ended_at": time.time()}
            write_manifest(run_dir, manifest.model_copy(update=failed))
            raise PreflightRefusedError(refused.code, refused.reason, refused.params)
        if not busy and not short:
            try:
                return gpu_lock.acquire(path, who=who, job=manifest.name, run_id=manifest.run_id,
                                        host_pid=os.getpid(), run_dir=_relative(run_dir, project_root))  # fmt: skip
            except gpu_lock.GpuLockError:
                pass  # another job took it between the check and here: wait for that one too
        if on_wait is not None:
            try:
                on_wait(_queue_info(since, busy=bool(busy), short=short[0] if short else None))
            except Exception:  # a broken UI callback must never stop the wait
                logger.exception("on_wait callback failed")
        if cancel is not None and cancel.wait(GPU_POLL_S):
            return manifest.model_copy(update={"status": RunStatus.CANCELLED, "ended_at": time.time()})
        if cancel is None:
            time.sleep(GPU_POLL_S)


def _wait_for_memory(
    manifest: RunManifest,
    run_dir: Path,
    est_peak_gb: float,
    *,
    cancel: threading.Event | None,
    on_wait: Callable[[dict[str, float]], None] | None,
    timeout_s: float,
    thresholds: MemThresholds,
) -> RunManifest | None:
    """Queue until the job fits the memory budget; the CANCELLED manifest if cancelled meanwhile, else None."""
    from vlm_pipeline.common.preflight import MemoryWaitTimeoutError, wait_for_memory

    since = time.time()

    def _report(info: dict[str, float]) -> None:
        if on_wait is not None:
            try:
                on_wait({**info, "since": since})
            except Exception:  # a broken UI callback must never stop the wait
                logger.exception("on_wait callback failed")

    try:
        fits = wait_for_memory(est_peak_gb, thresholds=thresholds, timeout_s=timeout_s, on_wait=_report, cancel=cancel)
    except MemoryWaitTimeoutError:
        failed = {"status": RunStatus.FAILED, "error_code": "SYS-002", "ended_at": time.time()}
        write_manifest(run_dir, manifest.model_copy(update=failed))
        raise
    if fits:
        return None
    final = manifest.model_copy(update={"status": RunStatus.CANCELLED, "ended_at": time.time()})
    write_manifest(run_dir, final)
    return final


def _relative(run_dir: Path, project_root: Path) -> str | None:
    """``run_dir`` relative to the project (for the GPU lock), None when it lives elsewhere."""
    root, path = project_root.resolve(), run_dir.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else None


def _checks_before_queueing(
    job: ContainerJob,
    manifest: RunManifest,
    project_root: Path,
    *,
    gate: bool,
    require_approved: bool,
    checks_images: bool = True,
) -> str | None:
    """Cheap refusals with no side effect, before the memory wait: returns the pinned image id."""
    # P017 B1 (bao-security): a run id that already had its job is refused before ANYTHING is written to its run dir
    # (the memory wait and run_job's failure path both write run.json): its history is never rewritten.
    if is_claimed(project_root, manifest.run_id):
        raise RunDirClaimedError(manifest.run_id)
    if job.narrow:  # an input the mount rules refuse fails here with SEC-001, not SYS-000 after the run dir exists
        _refuse_bad_inputs(job, project_root)
    if gate:  # gate/published runs must be reproducible before they take the GPU (ADR-002 A1)
        _check_gate(job, manifest, project_root)
    if not checks_images:  # vlm-runner gates and pins the image itself (compose: no docker CLI in the app)
        return None
    pinned_id = image_digest(job.image)  # the id the gate approves is the id that runs (no retag window)
    check_pinned_image(project_root, job.image, pinned_id, require_approved=require_approved)
    return pinned_id


def with_code_root(job: ContainerJob, project_root: Path) -> ContainerJob:
    """``job`` reading its code from ``$VLM_CODE_ROOT`` (data, outputs and the lock stay the project's).

    The shared working tree holds other people's unfinished code; a run that must be reproducible points this at a
    ``git worktree add --detach HEAD`` checkout. It must be a checkout of this project INSIDE the project (the
    worktrees live under .cache/ or .worktrees/), else ValueError. A job that names its own code_root keeps it.
    """
    raw = os.environ.get(CODE_ROOT_ENV, "").strip()
    if job.code_root is not None or not raw:
        return job
    code = Path(raw).resolve()
    root = project_root.resolve()
    if code == root:
        return job
    if not _is_team_worktree(root, code):
        raise ValueError(
            f"${CODE_ROOT_ENV} must be a git worktree checkout of this project (.cache/wt-*, .worktrees/*)"
        )
    return dataclasses.replace(job, code_root=code)


def _is_team_worktree(root: Path, code: Path) -> bool:
    """A real git worktree of this project where the team keeps them, never a folder a job can write (security C1)."""
    return is_team_worktree_dir(root, code) and git_state(code)[0] is not None


def _gpu_turn(
    job: ContainerJob, queue: tuple[RunManifest, Path, Path, float, float], *, take: bool, **wait: Any
) -> gpu_lock.LockHandle | RunManifest | None:
    """The GPU lock (or the cancelled manifest) for a job that holds it; None for one that never does."""
    if not take:
        if job.gpu:
            raise ValueError("a GPU job always holds the GPU lock")
        return None
    if job.egress:
        raise ValueError("a download job never holds the GPU lock")
    return _wait_for_gpu(*queue, **wait)  # (manifest, run_dir, lock path, est_peak_gb, ckpt_gb)


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
    require_approved_image: bool = True,
    on_wait: Callable[[dict[str, float]], None] | None = None,
    memory_wait_s: float = 1800.0,
    lock_wait_s: float = 12 * 3600.0,
    launcher: JobLauncher | None = None,
    take_gpu_lock: bool = True,
) -> RunManifest:
    """Run ``job`` end to end under the GPU lock; always writes a final run.json and releases the lock.

    ``take_gpu_lock=False`` only for jobs that never touch the GPU and must not wait behind a training run (catalog
    downloads: their own one-at-a-time lock, P018); CPU-heavy jobs such as a LoRA merge keep the GPU lock.

    ``require_approved_image`` is fail-closed: only a team probe passes False explicitly (``jobrun_cli
    --allow-unapproved-image``); every other job needs a ``bao-security APPROVED`` row in docker/IMAGES.lock.

    Not enough free memory (ADR-005 A1 budget) -> the job WAITS before taking the GPU lock (``wait_for_memory``),
    reporting ``on_wait({need_gb, avail_gb, waited_s, since})`` on every poll; cancel while waiting -> CANCELLED
    without starting a container; still no memory after ``memory_wait_s`` -> FAILED with SYS-002 (raised).
    """
    launcher = launcher or default_launcher()
    job = with_code_root(job, project_root)
    pinned_id = _checks_before_queueing(
        job, manifest, project_root, gate=gate, require_approved=require_approved_image,
        checks_images=getattr(launcher, "checks_images", True),
    )  # fmt: skip
    waited = _wait_for_memory(
        manifest,
        run_dir,
        est_peak_gb,
        cancel=cancel,
        on_wait=on_wait,
        timeout_s=memory_wait_s,
        thresholds=mem_thresholds,
    )
    if waited is not None:  # cancelled while waiting: nothing started, nothing to clean up
        return waited
    queue = (manifest, run_dir, lock_path or gpu_lock.default_lock_path(), est_peak_gb, ckpt_gb)
    taken = _gpu_turn(
        job, queue, take=take_gpu_lock, who=who, project_root=project_root, cancel=cancel, on_wait=on_wait,
        timeout_s=lock_wait_s, thresholds=mem_thresholds, memory_wait_s=memory_wait_s,
    )  # fmt: skip
    if isinstance(taken, RunManifest):  # cancelled while queued for the GPU
        return taken
    handle = taken
    container = f"vlm-{job.name}"
    try:
        # One job per run dir, ever (P017): the atomic claim is the real gate (the early check above only avoids
        # writes); taken under the lock, before the manifest says RUNNING, so a lost race writes nothing here.
        claim_run(project_root, manifest.run_id, by=who)
        commit, dirty = git_state(job.code_root or project_root)
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "pid": os.getpid(),
                "git_commit": manifest.git_commit or commit,
                "git_dirty": dirty if manifest.git_dirty is None else manifest.git_dirty,
                "container_image": manifest.container_image or job.image,
                "container_digest": manifest.container_digest or pinned_id,  # what the gate approved = what runs
                "container_started_at": time.time(),  # after the memory wait and the lock: no queueing in it
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
                    launcher.stop(container, STOP_GRACE_S)
                    return

        if cancel is not None:
            threading.Thread(target=_watch_cancel, name=f"cancel-{container}", daemon=True).start()
        log_path = run_dir / "logs" / "stdout.log"
        _, guard = run_guarded(
            lambda: launcher.run(
                job,
                project_root,
                lock_token=handle.info.token if handle else None,
                log_path=log_path,
                image_id=pinned_id,
            ),
            log_path,
            on_abort=lambda: launcher.stop(container, STOP_GRACE_S),
            hard_exit=lambda: launcher.kill(container),
            thresholds=mem_thresholds,
            # Publish the resolved stop level while the job runs, not only at the end: the E2E gate (bao-tester) waits
            # for MemAvailable >= max(16, abort of the lock holder + 4) and reads it from this run.json.
            on_guard_ready=lambda g: write_manifest(
                run_dir, manifest.model_copy(update={"memguard_abort_gb": round(g.thresholds.abort_gb, 2)})
            ),
        )
        done.set()
        state = launcher.state(container)
        runner_id = getattr(launcher, "last_image_id", None)  # vlm-runner reports the id it ran
        info = dataclasses.replace(
            state, preempted_by_memguard=guard.should_stop.is_set(), cancelled_by_executor=cancelled.is_set()
        )
        status, code = classify_exit(info)
        if status == RunStatus.COMPLETED and _stopped_by_signal(run_dir / "logs" / "stdout.log"):
            # AutoModel turns SIGTERM into a graceful stop with exit 0: not a finished run (stopped from outside).
            status, code = RunStatus.INTERRUPTED, None
        manifest = manifest.model_copy(
            update={
                "container_digest": runner_id or manifest.container_digest,
                "status": status,
                "error_code": code,
                "exit_code": info.exit_code,
                "oom_killed": info.oom_killed,
                "ended_at": time.time(),
                # Host-level estimate (includes CUDA on UMA, which the container cgroup misses) + shared-host noise.
                "peak_mem_gb": round(max(0.0, baseline_gb - guard.min_available_gb), 1),
                "host_min_available_gb": round(guard.min_available_gb, 1),
                "trainable_pct": _trainable_pct(run_dir / "logs" / "stdout.log"),
                "engine_warnings": _engine_warnings(run_dir / "logs" / "stdout.log"),
                "memguard_fast_fall": bool(guard.fast_fall_killed),
                "memguard_abort_gb": round(
                    guard.thresholds.abort_gb, 2
                ),  # resolved (80 % of the start-time free memory vs absolute)
            }
        )
        write_manifest(run_dir, manifest)
        return manifest
    except BaseException as exc:
        if isinstance(exc, RunDirClaimedError):
            raise  # that run dir belongs to the job that claimed it: never write its run.json
        if not manifest.status.is_terminal:  # runner itself failed: never leave "running" behind
            # Not launched because memory was already below abort -> MEM-001 (insufficient), not an unknown error.
            code = "MEM-001" if isinstance(exc, MemoryPreemptedBeforeStartError) else "SYS-000"
            write_manifest(
                run_dir,
                manifest.model_copy(update={"status": RunStatus.FAILED, "error_code": code, "ended_at": time.time()}),
            )
        raise
    finally:
        launcher.remove(container)
        if handle is not None:
            gpu_lock.release(handle)
