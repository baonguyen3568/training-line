"""Host side of the trial-inference node ("Chạy thử", ``serve.vllm``, M1b B-7; proposal 014).

One-shot container job through ``run_job`` (preflight, GPU lock, memory watchdog, run.json), never a live server.
The container side is ``vlm_pipeline.serve.trial``; this module resolves WHAT to load so the answer matches training:
- the catalog entry and pinned base snapshot of the model, also for a model produced by a fine-tune node (its run.json
  names the base): processor + generation_config always come from the base;
- a LoRA output (``adapter_config.json``) is loaded on top of the base, a full fine-tune export directly;
- the processor's min/max pixels from the run's training config (or the model's training template).
``engine=vllm`` runs vLLM's offline engine (``serve.vllm_runner``, one job, no server) for chat models in
``VLLM_PARITY_MODELS`` (vLLM<->HF parity passed, proposal 014 §2), without a LoRA adapter; anything else runs on HF and
the result says so (``engine_used`` + ``engine_fallback``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vlm_pipeline.app.uploads import UPLOADS_DIR, upload_path
from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_small_json
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.serve.config import InferenceParams
from vlm_pipeline.serve.trial import MAX_IMAGE_BYTES, TrialRequest, decoding_overrides

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker


class TrialImageError(VlmPipelineError):
    """The trial image is missing or unusable (reason: missing, not_a_file, too_large, too_many_pixels, unreadable)."""

    code: ClassVar[str] = "DATA-005"


def check_support(node_id: str, params: object, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Pure support rule of "Chạy thử", run at validate time and again before the job (no disk access).

    The only refusal that needs no file: no image chosen yet (DATA-005 missing). engine=vllm is never refused (when vLLM
    cannot serve the request it runs on HF and records why); every catalog family can answer. Missing files are
    runtime checks (``check_image``).
    """
    del model, inputs  # every catalog model and wiring the spec allows is supported
    if isinstance(params, InferenceParams) and not params.image:
        raise TrialImageError("no image", params={"node": node_id, "reason": "missing"})


def check_image(root: Path, node_id: str, upload_id: str) -> Path:
    """Host checks before the GPU lock (bao-security): a regular file, no symlink, under <root>/uploads/, <= 20 MB."""
    if not upload_id:
        raise TrialImageError("no image", params={"node": node_id, "reason": "missing"})
    uploads = (root / UPLOADS_DIR).resolve()
    path = upload_path(root, upload_id)
    try:
        st = path.lstat()
    except OSError:
        raise TrialImageError("image not found", params={"node": node_id, "reason": "missing"}) from None
    resolved = path.resolve()
    if path.is_symlink() or not path.is_file() or resolved.parent != uploads:
        raise TrialImageError("not an uploaded file", params={"node": node_id, "reason": "not_a_file"})
    if st.st_size > MAX_IMAGE_BYTES:
        raise TrialImageError("image too large", params={"node": node_id, "reason": "too_large"})
    return resolved


def _parents_in(root: Path, path: Path) -> list[Path]:
    """Ancestors of ``path`` up to (excluding) the project root: never read a run.json/config.yaml outside it."""
    top = root.resolve()
    return [p for p in path.resolve().parents if top in p.parents]


def base_of(worker: RealWorker, ref: ArtifactRef) -> tuple[CatalogEntry, Path]:
    """Catalog entry + host base snapshot of a model artifact.

    The catalog model itself, or the base recorded in the run.json of the fine-tune that produced it (one resolver:
    ``RealWorker.base_of``, shared with eval and bench).
    """
    entry = worker.base_of(ref)
    if "@" in ref.hash.split(":", 1)[0]:  # a catalog model: the artifact is the base snapshot
        return entry, worker.model_dir(ref)
    return entry, worker._catalog_model_path(entry).resolve()


def _pixels(worker: RealWorker, entry: CatalogEntry, ref: ArtifactRef) -> tuple[int, int]:
    """Processor min/max pixels exactly as the model was (or would be) trained."""
    import yaml

    model_path = worker.model_dir(ref)
    run_cfg = next(
        (p / "config.yaml" for p in _parents_in(worker.root, model_path) if (p / "config.yaml").is_file()), None
    )
    if run_cfg is None:
        from vlm_pipeline.train.recipe import RecipeError, template_path

        try:
            run_cfg = template_path(worker.root, entry)
        except RecipeError:  # no template: the processor's own defaults (Parse: fixed size)
            return 0, 0
    processor = (yaml.safe_load(run_cfg.read_text(encoding="utf-8")) or {}).get("processor") or {}
    return int(processor.get("min_pixels") or 0), int(processor.get("max_pixels") or 0)


def template_overrides(worker: RealWorker, entry: CatalogEntry) -> dict[str, Any]:
    """The training template's decode_overrides (001a §10.7), the same source eval uses; {} without a template."""
    import yaml

    from vlm_pipeline.common.decoding import decode_overrides_of
    from vlm_pipeline.train.recipe import RecipeError, template_path

    try:
        path = template_path(worker.root, entry)
    except RecipeError:  # no template for this model: keep the base decoding
        return {}
    return decode_overrides_of(yaml.safe_load(path.read_text(encoding="utf-8")) or {}).as_dict()


def request_inputs(worker: RealWorker, request: TrialRequest, *extra: str) -> tuple[Path, ...]:
    """What a narrowed trial/bench job reads (P017 step 6): every dir/file its request names, mounted read-only."""
    paths = [request.model_dir, request.generation_config_dir, request.processor_dir, request.adapter_dir,
             request.image, *extra]  # fmt: skip
    return tuple(worker.mount_input(p) for p in paths if p)


def build_request(worker: RealWorker, job: NodeJob, image: Path) -> tuple[TrialRequest, CatalogEntry]:
    params = job.params
    assert isinstance(params, InferenceParams)  # noqa: S101 - registry guarantees the params model
    ref = job.inputs["model"]
    entry, base = base_of(worker, ref)
    model_path = worker.model_dir(ref)
    adapter = model_path if (model_path / "adapter_config.json").is_file() else None
    min_px, max_px = _pixels(worker, entry, ref)
    family = "nemotron_parse" if entry.loss == "nemotron_parse" else "chat"
    request = TrialRequest(
        model_dir=worker.container_path(base if adapter else model_path),
        generation_config_dir=worker.container_path(base),
        processor_dir=worker.container_path(base),
        family=family,
        image=worker.container_path(image),
        prompt="" if entry.task_prompt_fixed else (params.prompt or "Describe this image."),
        max_new_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        adapter_dir=worker.container_path(adapter) if adapter else None,
        min_pixels=min_px,
        max_pixels=max_px,
        attn=entry.attn,
        decode_overrides=template_overrides(worker, entry),
    )
    return request, entry


# Catalog ids whose vLLM answers passed the vLLM<->HF parity gate (proposal 014 §2, tester): only these run on vLLM;
# every other request falls back to HF and says why (engine_used + engine_fallback). Empty until the gate passes.
VLLM_PARITY_MODELS: frozenset[str] = frozenset()


def vllm_unavailable(entry: CatalogEntry, request: TrialRequest) -> str | None:
    """Why this trial cannot run on vLLM yet (None = it can)."""
    if request.family != "chat":
        return "family"  # Nemotron-Parse: encoder-decoder with a forced prefix, HF only (P014 §2)
    if request.adapter_dir:
        return "lora_adapter"  # runtime LoRA loading is off by design (bao-security); merge the adapter first
    if entry.id not in VLLM_PARITY_MODELS:
        return "parity_pending"
    return None


def vllm_sampling(worker: RealWorker, request: TrialRequest) -> tuple[dict[str, float], dict[str, Any]]:
    """Sampling for vLLM = base generation_config.json -> template decode_overrides -> node params (the HF order).

    Returns (sampling for the batch item, the resolved decoding to record). Greedy stays greedy: temperature 0.
    """
    path = worker.host_path(request.generation_config_dir) / "generation_config.json"
    base: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    resolved = {**base, **request.decode_overrides, **decoding_overrides(request)}
    sampling: dict[str, float] = {}
    if resolved.get("do_sample"):
        for key in ("temperature", "top_p", "top_k"):
            if resolved.get(key) is not None:
                sampling[key] = float(resolved[key])
    if resolved.get("repetition_penalty") not in (None, 1.0):
        sampling["repetition_penalty"] = float(resolved["repetition_penalty"])
    source = "base generation_config.json" if base else "model defaults (no generation_config.json)"
    node = {k: v for k, v in decoding_overrides(request).items() if k != "use_cache"}  # an HF generate() key only
    return sampling, {"_source": source, "_template": dict(request.decode_overrides), **node}


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write result.json in the job's run dir from the HOST.

    A name the job may have planted (a link) is replaced, never written through: unlink, then O_EXCL | O_NOFOLLOW.
    """
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1))


def _run_vllm(
    worker: RealWorker, job: NodeJob, request: TrialRequest, image: Path, manifest: RunManifest
) -> dict[str, Any]:
    from vlm_pipeline.serve.vllm_runner import run_vllm_batch

    params = job.params
    assert isinstance(params, InferenceParams)  # noqa: S101
    sampling, record = vllm_sampling(worker, request)
    item = {"id": "trial", "image": str(image), "prompt": request.prompt, "max_tokens": request.max_new_tokens,
            "sampling": sampling}  # fmt: skip
    res = run_vllm_batch(
        worker.root, worker.host_path(request.model_dir), [item], who=worker.who,
        mem_frac=params.gpu_memory_utilization, max_model_len=params.max_model_len,
        min_pixels=request.min_pixels, max_pixels=request.max_pixels,
        cancel=job.cancel, on_wait=job.on_wait, run_dir=job.run_dir, manifest=manifest,
    )  # fmt: skip
    try:
        row = res.rows()[0] if res.ok else {}
        engine = res.engine() if res.ok else {}
    except (OSError, ValueError):  # a job-made link / FIFO / garbage: no answer
        row, engine = {}, {}
    if "error" in row:
        reason = str(row["error"].get("reason", "unreadable"))
        raise TrialImageError("image rejected in the container", params={"node": job.node_id, "reason": reason})
    if "text" not in row:
        from vlm_pipeline.common.run_log import log_tail

        raise ContainerExit(
            75 if res.status == RunStatus.INTERRUPTED else 1,
            cancelled_by_executor=res.status == RunStatus.CANCELLED,
            log_tail=log_tail(res.run_dir),
        )
    keep = ("vllm", "quantization", "dtype", "gpu_memory_utilization", "max_model_len", "container_digest")
    return {
        "engine_used": "vllm",
        "engine_requested": "vllm",
        "family": request.family,
        "prompt_token_count": row.get("prompt_tokens"),
        "prompt_ids_sha": row.get("prompt_ids_sha"),
        "decoding": row.get("decoding"),
        "overrides": record,
        "text": str(row["text"]).strip(),
        "n_tokens": row.get("gen_tokens"),
        "truncated": row.get("finish_reason") == "length",
        "engine": {k: engine.get(k) for k in keep},
    }


def est_peak_gb(entry: CatalogEntry) -> float:
    """bf16 weights + activations/KV for one image: a preflight estimate, memguard enforces the real peak."""
    return round(entry.params_b * 2 * 1.25 + 6, 1)


def run_trial(worker: RealWorker, job: NodeJob) -> NodeResult:
    params = job.params
    assert isinstance(params, InferenceParams)  # noqa: S101
    image = check_image(worker.root, job.node_id, params.image)
    request, entry = build_request(worker, job, image)
    run_dir = job.run_dir.resolve()
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").unlink(missing_ok=True)  # a stale answer must never be shown for this run
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(asdict(request), indent=1), encoding="utf-8")
    run_id = run_dir.name[:120]
    manifest = RunManifest(
        run_id=run_id,
        name=f"{job.spec.type_id} {entry.display_name}",
        stage="serve",
        task="trial",
        base_model=entry.repo,
        base_model_revision=entry.revision,
        container_image=worker.image,
    )
    container = ContainerJob(
        name=run_id,
        image=worker.image,
        argv=("python", "-m", "vlm_pipeline.serve.trial", worker.container_path(request_path)),
        # bao-security: remote model code runs on a user's image -> the project is read-only, only this run's dir is
        # writable (result.json), and caches the job would write to the project go to the container's /tmp.
        read_only_project=True,
        rw_dirs=(run_dir,),
        env={"HF_MODULES_CACHE": "/tmp/hf_modules", "TRITON_CACHE_DIR": "/tmp/triton"},  # noqa: S108
        narrow=True,
        ro_inputs=request_inputs(worker, request),
    )
    reason = vllm_unavailable(entry, request) if params.engine == "vllm" else None
    if params.engine == "vllm" and reason is None:
        answer = _run_vllm(worker, job, request, image, manifest)
        _write_result(run_dir / "result.json", answer)
        return _endpoint(job, run_dir / "result.json")
    try:
        final = run_job(
            container,
            manifest,
            run_dir,
            project_root=worker.root,
            who=worker.who,
            est_peak_gb=est_peak_gb(entry),
            cancel=job.cancel,
            require_approved_image=True,
            on_wait=job.on_wait,  # not enough free memory: the node shows "waiting for memory" (ADR-005 A1)
        )
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc), params={"need_gb": est_peak_gb(entry)}) from exc
    result_path = run_dir / "result.json"
    try:  # written by the job (remote model code): no link, no FIFO, bounded (bao-security B1 class)
        raw = read_small_json(result_path)
    except (OSError, ValueError):
        raw = {}
    result: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if "error" in result:
        reason = str(result["error"].get("reason", "unreadable"))
        raise TrialImageError("image rejected in the container", params={"node": job.node_id, "reason": reason})
    if final.status != RunStatus.COMPLETED or "text" not in result:
        from vlm_pipeline.common.run_log import log_tail

        raise ContainerExit(
            75 if final.status == RunStatus.INTERRUPTED else (final.exit_code or 1),
            oom_killed=bool(final.oom_killed),
            cancelled_by_executor=final.status == RunStatus.CANCELLED,
            log_tail=log_tail(run_dir),
        )
    if params.engine == "vllm":  # proposal 014 §2: asked for vLLM, ran on HF, and says why (never silently)
        result.update(engine_requested="vllm", engine_fallback=reason)
        _write_result(result_path, result)
    return _endpoint(job, result_path)


def _endpoint(job: NodeJob, result_path: Path) -> NodeResult:
    port = next(iter(job.spec.outputs))
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()  # content identity, like every other artifact
    return NodeResult(outputs={port: ArtifactRef(type="Endpoint", path=str(result_path), hash=f"sha256:{digest}")})


__all__ = [
    "UPLOADS_DIR",
    "VLLM_PARITY_MODELS",
    "TrialImageError",
    "base_of",
    "build_request",
    "check_image",
    "check_support",
    "est_peak_gb",
    "run_trial",
    "template_overrides",
    "upload_path",
    "vllm_sampling",
    "vllm_unavailable",
]
