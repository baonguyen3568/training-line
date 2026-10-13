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
import secrets
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
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest
from vlm_pipeline.serve.vllm_batch import CONTAINER_ENV, ENGINE_NAME, OUT_NAME, BatchItem, BatchRequest

VLLM_IMAGE = "vlm-vllm:0.31.0"  # docker/IMAGES.lock, bao-security APPROVED 2026-10-13
DEFAULT_MEM_FRAC = 0.3
CPU_SIDE_GB = 4.0  # engine process + tokenizer/processor + image decode on the CPU side (UMA: same pool)
MIN_KV_GB = 2.0  # below this the engine starts but cannot hold one 8k request with an image
MAX_TOKENS_LIMIT = 8192
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


@dataclass(frozen=True)
class BatchResult:
    run_dir: Path
    status: RunStatus
    error_code: str | None
    container_digest: str | None

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED and (self.run_dir / OUT_NAME).is_file()

    @property
    def out_path(self) -> Path:
        return self.run_dir / OUT_NAME

    def rows(self) -> list[dict[str, Any]]:
        """One dict per request, in request order.

        {id, text, finish_reason, prompt_tokens, gen_tokens, prompt_sha, prompt_ids_sha}, or {id, error: {code, reason}}
        for a refused image.
        """
        with self.out_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def engine(self) -> dict[str, Any]:
        """engine.json + the image id that really ran (run.json container_digest)."""
        engine: dict[str, Any] = json.loads((self.run_dir / ENGINE_NAME).read_text(encoding="utf-8"))
        return {**engine, "image": VLLM_IMAGE, "container_digest": self.container_digest}


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
) -> EngineBudget:
    """The engine's ``gpu_memory_utilization`` for this job under the 80 %-of-free rule (pure).

    Wanted: ``mem_frac`` x total. Allowed now: ``max_free_frac`` x available - CPU side. Needed at least: weights + 10 %
    + a minimal KV cache. The engine gets min(wanted, allowed) but never less than needed; when needed > allowed the
    estimate says so and ``run_job`` queues the job until memory frees up. Needed > wanted is a caller error.
    """
    if not 0.0 < mem_frac <= 1.0:
        raise BatchRequestError(f"mem_frac must be in (0, 1], got {mem_frac}")
    wanted = mem_frac * total_gb
    needed = weights * 1.1 + MIN_KV_GB
    if needed > wanted:
        raise BatchRequestError(
            f"the model needs {needed:.1f} GB but mem_frac {mem_frac} gives the engine {wanted:.1f} GB"
        )
    allowed = thresholds.max_free_frac * available_gb - CPU_SIDE_GB
    engine = max(needed, min(wanted, allowed))
    util = round(engine / total_gb, 3)
    return EngineBudget(
        gpu_memory_utilization=util, engine_gb=round(engine, 1), est_peak_gb=round(engine + CPU_SIDE_GB, 1)
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
        items.append(BatchItem(id=rid, image=image, prompt=str(raw.get("prompt", "")), max_tokens=max_tokens))
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
    extra: Mapping[str, Any] | None = None,
    cancel: threading.Event | None = None,
    on_wait: Callable[[dict[str, float]], None] | None = None,
    thresholds: MemThresholds = DEFAULT_THRESHOLDS,
    memory: tuple[float, float] | None = None,
    launcher: JobLauncher | None = None,
) -> BatchResult:
    """Answer ``requests`` (jsonl path or dicts: id, image, prompt, max_tokens) greedily with vLLM on ``model_dir``.

    Everything is checked before a run dir exists (a refused batch leaves nothing behind). ``memory`` = (total,
    available) GB for tests; by default read now (``host_memory``).
    """
    root = root.resolve()
    model = model_dir.resolve()
    if not model.is_relative_to(root) or not (model / "config.json").is_file():
        raise BatchRequestError("model_dir must be an HF model dir inside the project")
    if bool(min_pixels) != bool(max_pixels) or min_pixels > max_pixels:
        raise BatchRequestError("min_pixels / max_pixels go together, min <= max")
    if weights_gb(model) == 0:
        raise BatchRequestError("model_dir has no *.safetensors weights (other formats are never loaded)")
    items, image_mounts = build_items(root, read_requests(requests) if isinstance(requests, Path) else requests)
    total, available = memory or host_memory()
    budget = engine_budget(weights_gb(model), total_gb=total, available_gb=available, mem_frac=mem_frac,
                           thresholds=thresholds)  # fmt: skip

    run_id = f"{time.strftime('%Y%m%d-%H%M')}-{tag}-{secrets.token_hex(2)}"
    run_dir = root / "outputs" / "runs" / run_id
    (run_dir / "logs").mkdir(parents=True)
    request = BatchRequest(
        model_dir=_container_path(root, model),
        items=items,
        gpu_memory_utilization=budget.gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=quantization,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        enable_thinking=enable_thinking,
        extra={**(extra or {}), "engine_gb": budget.engine_gb, "est_peak_gb": budget.est_peak_gb},
    )
    request_path = run_dir / "batch.json"
    request_path.write_text(request.to_json(), encoding="utf-8")
    manifest = RunManifest(run_id=run_id, name=f"vllm batch {model.name}", stage="eval", task="vllm_batch",
                           container_image=VLLM_IMAGE, seed=0)  # fmt: skip
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
