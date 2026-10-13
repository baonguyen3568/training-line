"""Host side of the trial-inference node ("Chạy thử", ``serve.vllm``, M1b B-7; proposal 014).

One-shot container job through ``run_job`` (preflight, GPU lock, memory watchdog, run.json), never a live server.
The container side is ``vlm_pipeline.serve.trial``; this module resolves WHAT to load so the answer matches training:
- the catalog entry and pinned base snapshot of the model, also for a model produced by a fine-tune node (its run.json
  names the base): processor + generation_config always come from the base;
- a LoRA output (``adapter_config.json``) is loaded on top of the base, a full fine-tune export directly;
- the processor's min/max pixels from the run's training config (or the model's training template).
``engine=vllm`` runs on HF until the vLLM image is probed and parity-tested (proposal 014 §2); ``engine_used`` says so.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vlm_pipeline.app.uploads import UPLOADS_DIR, upload_path
from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.serve.config import InferenceParams
from vlm_pipeline.serve.trial import MAX_IMAGE_BYTES, TrialRequest

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker


class TrialImageError(VlmPipelineError):
    """The trial image is missing or unusable (reason: missing, not_a_file, too_large, too_many_pixels, unreadable)."""

    code: ClassVar[str] = "DATA-005"


def check_support(node_id: str, params: object, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Pure support rule of "Chạy thử", run at validate time and again before the job (no disk access).

    The only refusal that needs no file: no image chosen yet (DATA-005 missing). engine=vllm is not refused (it runs on
    HF and records engine_used); every catalog family can answer. Missing files are runtime checks (``check_image``).
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
    )
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
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    if "error" in result:
        reason = str(result["error"].get("reason", "unreadable"))
        raise TrialImageError("image rejected in the container", params={"node": job.node_id, "reason": reason})
    if final.status != RunStatus.COMPLETED or "text" not in result:
        from vlm_pipeline.app.workers import _tail

        raise ContainerExit(
            75 if final.status == RunStatus.INTERRUPTED else (final.exit_code or 1),
            oom_killed=bool(final.oom_killed),
            cancelled_by_executor=final.status == RunStatus.CANCELLED,
            log_tail=_tail(run_dir / "logs" / "stdout.log"),
        )
    if params.engine == "vllm":
        result["engine_requested"] = "vllm"  # proposal 014 §2: HF until the vLLM image passes probe + parity
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    port = next(iter(job.spec.outputs))
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()  # content identity, like every other artifact
    return NodeResult(outputs={port: ArtifactRef(type="Endpoint", path=str(result_path), hash=f"sha256:{digest}")})


__all__ = [
    "UPLOADS_DIR",
    "TrialImageError",
    "base_of",
    "build_request",
    "check_image",
    "check_support",
    "est_peak_gb",
    "run_trial",
    "template_overrides",
    "upload_path",
]
