"""Host side of the offline vLLM batch (B7-3 step 1): ``run_vllm_batch`` = one container job through ``run_job``.

What tester's FP8 gate (bf16 vs FP8 OCRBench on the same engine) and R3 need, with bao-security's default for eval and
bench: vLLM's offline ``LLM`` in ``vlm-vllm:0.31.0`` (APPROVED), ``--network none``, no server and no port. The job is
narrowed (P017 step 6): the only writable dir is its run dir; the model's repo dir and the images' dirs are mounted
read-only. Preflight, the memory queue (``on_wait``), the GPU lock and memguard come from ``run_job``; the container is
removed whatever happens (also on an exception or cancel).

Memory (user rule, ADR-005 A1): the engine takes ``gpu_memory_utilization`` x total. It asks for ``mem_frac`` (default
0.3) of the total, shrunk to fit 80 % of what is free now (minus the process's CPU side), never below what the weights
plus a minimal KV cache need: then the job waits in the queue for memory instead of starting starved.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_pipeline.common.gpu_mem import memory_model, query_gpus
from vlm_pipeline.common.host_paths import hub_repo_dir
from vlm_pipeline.common.jobrun import CONTAINER_ROOT, ContainerJob, run_job
from vlm_pipeline.common.launcher import JobLauncher
from vlm_pipeline.common.memguard import DEFAULT_THRESHOLDS, MemThresholds, read_meminfo
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_small_json, write_manifest
from vlm_pipeline.serve.vllm_batch import (
    CONTAINER_ENV,
    ENGINE_NAME,
    OUT_NAME,
    SAMPLING_KEYS,
    BatchItem,
    BatchRequest,
)

VLLM_IMAGE = "vlm-vllm:0.31.0"  # docker/IMAGES.lock, bao-security APPROVED 2026-10-13
DEFAULT_MEM_FRAC = 0.3
# Outside the engine's pool, on the CPU side (UMA: the same memory): engine processes, tokenizer/processor, image
# decode, kernel JIT, and the weights' host copy while loading. Measured on GB10 2026-10-13 (peak MemAvailable drop,
# engine 36.5 GB): Qwen3.5-0.8B bf16 39.4 / FP8 46.1, Qwen3.5-2B bf16 53.7 / FP8 46.2 GB -> base + per weight GB.
CPU_SIDE_BASE_GB = 8.0
CPU_SIDE_PER_WEIGHT_GB = 2.5


def cpu_side_gb(weights: float) -> float:
    return CPU_SIDE_BASE_GB + CPU_SIDE_PER_WEIGHT_GB * weights


MIN_KV_GB = 2.0  # below this the engine starts but cannot hold one 8k request with an image
# Activations + CUDA graphs-free workspace + multimodal encoder cache outside the KV pool (eager mode). Measured on GB10
# (Qwen3.5-0.8B bf16/FP8, 2026-10-13): weights + KV + this left > 1 GB unused at mem_util 0.3.
ACT_RESERVE_GB = 3.0
# KV pool when the caller does not pin one. An eval batch (chunk 64, prompts ~1-2k tokens incl. the image, greedy
# answers <= 1k) needs a few GB on <= 2B models (Qwen3.5-0.8B: ~17 KB/token -> 64 x 3k tokens ~ 3.3 GB); the old
# pool sized to mem_frac (30+ GB) only raised the peak on a shared host (research). Short of it vLLM preempts, slower
# but correct, and engine.json reports num_preemptions.
DEFAULT_KV_GB = 8.0
MAX_TOKENS_LIMIT = 8192
MAX_LOGPROBS = 5
# bao-security S2: images are mounted one file at a time (no sibling files exposed); only a big batch from one dir
# outside uploads/ mounts that dir (docker has no cheap way to bind 500 files).
DIR_MOUNT_MIN = 200


class BatchRequestError(ValueError):
    """The batch itself is malformed (ids, paths, limits): refused before anything runs."""


@dataclass(frozen=True)
class EngineBudget:
    gpu_memory_utilization: float
    engine_gb: float
    est_peak_gb: float  # engine + CPU side: what preflight / the memory queue check
    kv_cache_gb: float  # pinned KV cache: engine - weights x 1.1 - activations (no profiling on a shared UMA host)


@dataclass(frozen=True)
class BatchResult:
    run_dir: Path
    status: RunStatus
    error_code: str | None
    container_digest: str | None

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED and (self.run_dir / OUT_NAME).exists()

    @property
    def out_path(self) -> Path:
        return self.run_dir / OUT_NAME

    def rows(self) -> list[dict[str, Any]]:
        """One dict per request, in request order.

        {id, text, finish_reason, prompt_tokens, gen_tokens, prompt_sha, prompt_ids_sha}, or {id, error: {code, reason}}
        for a refused image.
        """
        return [json.loads(line) for line in _read_job_text(self.out_path).splitlines() if line.strip()]

    def engine(self) -> dict[str, Any]:
        """engine.json + the image id that really ran (run.json container_digest)."""
        raw = read_small_json(self.run_dir / ENGINE_NAME)  # the job wrote it: no link, no FIFO, bounded
        engine: dict[str, Any] = raw if isinstance(raw, dict) else {}
        return {**engine, "image": VLLM_IMAGE, "container_digest": self.container_digest}


OUT_MAX_BYTES = 64 * 1024 * 1024  # 500 answers x 4096 tokens fit with a wide margin


def _read_job_text(path: Path, max_bytes: int = OUT_MAX_BYTES) -> str:
    """Text of a file the job wrote in its run dir: never through a link, never a FIFO, bounded (bao-security B1)."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            raise ValueError(f"{path.name}: not a regular file of at most {max_bytes} bytes")
        return f.read(max_bytes + 1).decode("utf-8")


def weights_gb(model_dir: Path) -> float:
    """Size of the checkpoint's weight files (HF snapshot links followed)."""
    return sum(p.stat().st_size for p in model_dir.glob("*.safetensors")) / 2**30


def engine_budget(
    weights: float,
    *,
    total_gb: float,
    available_gb: float,
    mem_frac: float = DEFAULT_MEM_FRAC,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    kv_cache_gb: float | None = None,
) -> EngineBudget:
    """Engine size = weights x 1.1 + activations + a pinned KV cache, under the 80 %-of-free rule (pure).

    KV: ``kv_cache_gb`` when given (a gate passes the SAME value to both runs of a pair, tester), else
    ``DEFAULT_KV_GB`` (research: eval batches use a few GB; a pool sized to mem_frac only raised the peak), shrunk to
    fit what is allowed now but never below ``MIN_KV_GB``. The engine never exceeds ``mem_frac`` x total (caller
    error). A given KV is never shrunk: when it does not fit now, ``run_job`` queues the job until memory frees up.
    """
    if not 0.0 < mem_frac <= 1.0:
        raise BatchRequestError(f"mem_frac must be in (0, 1], got {mem_frac}")
    if kv_cache_gb is not None and not MIN_KV_GB <= kv_cache_gb <= total_gb:
        raise BatchRequestError(f"kv_cache_gb must be in [{MIN_KV_GB}, {total_gb:.0f}], got {kv_cache_gb}")
    fixed = weights * 1.1 + ACT_RESERVE_GB
    wanted = mem_frac * total_gb
    kv = DEFAULT_KV_GB if kv_cache_gb is None else kv_cache_gb
    if fixed + (MIN_KV_GB if kv_cache_gb is None else kv) > wanted:
        raise BatchRequestError(
            f"the model needs {fixed + kv:.1f} GB but mem_frac {mem_frac} gives the engine {wanted:.1f} GB"
        )
    cpu = cpu_side_gb(weights)
    if kv_cache_gb is None:
        allowed = thresholds.max_free_frac * available_gb - cpu - fixed
        kv = max(MIN_KV_GB, min(kv, wanted - fixed, allowed))
    engine = fixed + kv
    return EngineBudget(
        gpu_memory_utilization=round(engine / total_gb, 3),
        engine_gb=round(engine, 1),
        est_peak_gb=round(engine + cpu, 1),
        kv_cache_gb=round(kv, 2),
    )


def host_memory() -> tuple[float, float]:
    """(total, available) GB of the pool the engine allocates from: VRAM on a discrete GPU, the host pool on UMA."""
    gpus = query_gpus()
    if memory_model(gpus) == "discrete" and gpus[0].total_gb and gpus[0].free_gb is not None:
        return gpus[0].total_gb, gpus[0].free_gb
    snap = read_meminfo()
    return snap.mem_total_gb, snap.mem_available_gb


def _container_path(root: Path, path: Path) -> str:
    return f"{CONTAINER_ROOT}/{path.relative_to(root).as_posix()}"


def _image_mounts(root: Path, images: list[Path]) -> list[Path]:
    by_dir: dict[Path, list[Path]] = {}
    for path in images:
        by_dir.setdefault(path.parent, []).append(path)
    mounts: list[Path] = []
    for parent, files in sorted(by_dir.items()):
        whole_dir = len(files) >= DIR_MOUNT_MIN and parent.relative_to(root).parts[:1] != ("uploads",)
        mounts += [parent] if whole_dir else sorted(set(files))
    return mounts


# Accepted ranges, checked on the host before any job (bao-security): NaN / inf / out of range never reach the GPU.
SAMPLING_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 2.0),
    "top_p": (1e-6, 1.0),
    "top_k": (-1, 1000),  # -1 = off; an integer
    "repetition_penalty": (0.5, 2.0),
}


def _checked_sampling(rid: str, raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise BatchRequestError(f"request {rid}: sampling must be a mapping")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key not in SAMPLING_KEYS:
            raise BatchRequestError(f"request {rid}: unknown sampling key {key!r}")
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise BatchRequestError(f"request {rid}: sampling {key} must be a finite number")
        low, high = SAMPLING_RANGES[key]
        if not low <= value <= high or (key == "top_k" and (value != int(value) or value == 0)):
            raise BatchRequestError(f"request {rid}: sampling {key}={value} out of range")
        out[key] = float(value)
    return out


def build_items(root: Path, requests: Iterable[Mapping[str, Any]]) -> tuple[list[BatchItem], list[Path]]:
    """Checked items (container paths) + the image files (or, for a big batch, dirs) to mount.

    Refuses duplicate ids, paths outside the project, symlinked images and silly limits: nothing runs on a malformed
    batch.
    """
    items: list[BatchItem] = []
    images: list[Path] = []
    seen: set[str] = set()
    for n, raw in enumerate(requests):
        rid = str(raw.get("id", ""))
        if not rid or rid in seen:
            raise BatchRequestError(f"request {n}: id {rid!r} is empty or duplicated")
        seen.add(rid)
        max_tokens = int(raw.get("max_tokens", 256))
        if not 1 <= max_tokens <= MAX_TOKENS_LIMIT:
            raise BatchRequestError(f"request {rid}: max_tokens {max_tokens} not in [1, {MAX_TOKENS_LIMIT}]")
        image = ""
        if raw.get("image"):
            given = Path(str(raw["image"]))
            path = given if given.is_absolute() else root / given
            if path.is_symlink() or not path.is_file():
                raise BatchRequestError(f"request {rid}: image is not a regular file")
            path = path.resolve()
            if not path.is_relative_to(root):
                raise BatchRequestError(f"request {rid}: image is outside the project")
            images.append(path)
            image = _container_path(root, path)
        sampling = _checked_sampling(rid, raw.get("sampling") or {})
        items.append(
            BatchItem(id=rid, image=image, prompt=str(raw.get("prompt", "")), max_tokens=max_tokens, sampling=sampling)
        )
    if not items:
        raise BatchRequestError("empty batch")
    return items, _image_mounts(root, images)


def read_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_vllm_batch(
    root: Path,
    model_dir: Path,
    requests: Path | Iterable[Mapping[str, Any]],
    *,
    who: str,
    tag: str = "vllm-batch",
    quantization: str | None = None,
    max_model_len: int = 8192,
    mem_frac: float = DEFAULT_MEM_FRAC,
    min_pixels: int = 0,
    max_pixels: int = 0,
    enable_thinking: bool = False,
    logprobs: int = 0,
    json_schema: Mapping[str, Any] | None = None,
    kv_cache_gb: float | None = None,
    extra: Mapping[str, Any] | None = None,
    cancel: threading.Event | None = None,
    on_wait: Callable[[dict[str, float]], None] | None = None,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    memory: tuple[float, float] | None = None,
    launcher: JobLauncher | None = None,
    run_dir: Path | None = None,
    manifest: RunManifest | None = None,
) -> BatchResult:
    """Answer ``requests`` (jsonl path or dicts: id, image, prompt, max_tokens) greedily with vLLM on ``model_dir``.

    Everything is checked before a run dir exists (a refused batch leaves nothing behind). ``memory`` = (total,
    available) GB for tests; by default read now (``host_memory``). A graph node passes its own ``run_dir`` (the
    executor's attempt dir, already claimed) and ``manifest`` (name, stage, base model); a CLI/QA caller gets a fresh
    ``outputs/runs/<id>``.
    """
    root = root.resolve()
    model = model_dir.resolve()
    if not model.is_relative_to(root) or not (model / "config.json").is_file():
        raise BatchRequestError("model_dir must be an HF model dir inside the project")
    if not 0 <= logprobs <= MAX_LOGPROBS:
        raise BatchRequestError(f"logprobs must be in [0, {MAX_LOGPROBS}]")
    if bool(min_pixels) != bool(max_pixels) or min_pixels > max_pixels:
        raise BatchRequestError("min_pixels / max_pixels go together, min <= max")
    if weights_gb(model) == 0:
        raise BatchRequestError("model_dir has no *.safetensors weights (other formats are never loaded)")
    items, image_mounts = build_items(root, read_requests(requests) if isinstance(requests, Path) else requests)
    total, available = memory or host_memory()
    budget = engine_budget(weights_gb(model), total_gb=total, available_gb=available, mem_frac=mem_frac,
                           thresholds=thresholds, kv_cache_gb=kv_cache_gb)  # fmt: skip

    if run_dir is None:
        run_id = f"{time.strftime('%Y%m%d-%H%M')}-{tag}-{secrets.token_hex(2)}"
        run_dir = root / "outputs" / "runs" / run_id
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(root / "outputs"):
        raise BatchRequestError("run_dir must be under outputs/")
    run_id = manifest.run_id if manifest is not None else run_dir.name[:120]
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    request = BatchRequest(
        model_dir=_container_path(root, model),
        items=items,
        gpu_memory_utilization=budget.gpu_memory_utilization,
        kv_cache_gb=budget.kv_cache_gb,
        max_model_len=max_model_len,
        quantization=quantization,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        enable_thinking=enable_thinking,
        logprobs=logprobs,
        json_schema=dict(json_schema) if json_schema is not None else None,
        extra={**(extra or {}), "engine_gb": budget.engine_gb, "est_peak_gb": budget.est_peak_gb},
    )
    request_path = run_dir / "batch.json"
    request_path.write_text(request.to_json(), encoding="utf-8")
    if manifest is None:
        manifest = RunManifest(run_id=run_id, name=f"vllm batch {model.name}", stage="eval", task="vllm_batch",
                               container_image=VLLM_IMAGE, seed=0)  # fmt: skip
    manifest = manifest.model_copy(update={"container_image": VLLM_IMAGE})
    write_manifest(run_dir, manifest)
    job = ContainerJob(
        name=run_id,
        image=VLLM_IMAGE,
        entrypoint="python3",
        argv=("-m", "vlm_pipeline.serve.vllm_batch", _container_path(root, request_path)),
        read_only_project=True,
        rw_dirs=(run_dir,),
        narrow=True,
        ro_inputs=(hub_repo_dir(root, model), *image_mounts),
        env=dict(CONTAINER_ENV),
    )
    final = run_job(job, manifest, run_dir, project_root=root, who=who, est_peak_gb=budget.est_peak_gb,
                    cancel=cancel, on_wait=on_wait, mem_thresholds=thresholds, launcher=launcher)  # fmt: skip
    return BatchResult(run_dir=run_dir, status=final.status, error_code=final.error_code,
                       container_digest=final.container_digest)  # fmt: skip


__all__ = [
    "VLLM_IMAGE",
    "BatchRequestError",
    "BatchResult",
    "EngineBudget",
    "build_items",
    "engine_budget",
    "host_memory",
    "read_requests",
    "run_vllm_batch",
    "weights_gb",
]
