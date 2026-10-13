"""Node ``quantize.modelopt_ptq`` for the real worker (B-4): FP8 PTQ of a native VLM in the training image.

Wired into ``RealWorker._handlers`` as ``functools.partial(run_quantize, self)`` (same pattern as the trial worker).
The container runs ``vlm_pipeline.quantize.ptq``; the job writes only its own run dir (project read-only).
Scope, memory estimate and argv: ``vlm_pipeline.quantize.scope``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vlm_pipeline.common.errors import ExitCode, InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.run_log import log_tail
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_small_json
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.quantize.config import QuantizeParams
from vlm_pipeline.quantize.scope import QuantizeNotSupportedError, check_supported, est_peak_gb, ptq_argv

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker


def run_quantize(worker: RealWorker, job: NodeJob) -> NodeResult:
    params = job.params
    assert isinstance(params, QuantizeParams)  # noqa: S101
    model = job.inputs["model"]
    entry = worker.base_of(model)
    check_supported(entry, params, job.node_id)
    calib = job.inputs.get("calib_data")
    if calib is None:
        raise QuantizeNotSupportedError(
            "calibration data required",
            params={"node": job.node_id, "node_type": job.spec.type_id, "reason": "calib_data_missing"},
        )
    data = worker.dataset_input(calib)
    if data.records is not None:
        raise QuantizeNotSupportedError(
            "local folders are not supported for calibration yet",
            params={"node": job.node_id, "node_type": job.spec.type_id, "reason": "local_calib"},
        )
    run_dir = job.run_dir.resolve()
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    out = run_dir / "ptq"
    run_id = run_dir.name[:120]
    manifest = RunManifest(
        run_id=run_id, name=f"{job.spec.type_id} {entry.display_name}", stage="quantize", task=params.format,
        base_model=entry.repo, base_model_revision=entry.revision, dataset=data.hash, container_image=worker.image,
    )  # fmt: skip
    container = ContainerJob(
        name=run_id,
        image=worker.image,
        argv=ptq_argv(
            model=worker.container_path(worker.model_dir(model)),
            data=data.path,
            out=worker.container_path(out),
            fmt=params.format,
            calib_size=params.calib_size,
        ),
        read_only_project=True,  # writes only its run dir; JIT caches go there too (jobrun.compile_cache_env)
        rw_dirs=(run_dir,),
        narrow=True,
        ro_inputs=(worker.mount_input(worker.model_dir(model)), worker.mount_input(data.path)),
    )
    need = est_peak_gb(entry, params.calib_size)
    try:
        final = run_job(
            container, manifest, run_dir, project_root=worker.root, who=worker.who, est_peak_gb=need,
            ckpt_gb=entry.disk_gb, cancel=job.cancel, require_approved_image=True, on_wait=job.on_wait,
        )  # fmt: skip
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc), params={"need_gb": need}) from exc
    parity = _parity(run_dir / "ptq" / "report.json")
    parity_failed = final.exit_code == ExitCode.QUANT_PARITY_FAIL and final.status == RunStatus.FAILED
    fail = _parity_params(job.node_id, parity, "fail") if parity and parity.get("verdict") == "fail" else None
    if parity_failed and fail is not None:  # broken: never handed over (without numbers: the generic exit below)
        raise QuantParityFailError("quantization parity fail", params=fail)
    if final.status != RunStatus.COMPLETED:
        raise ContainerExit(
            final.exit_code or 1,
            oom_killed=bool(final.oom_killed),
            cancelled_by_executor=final.status == RunStatus.CANCELLED,
            log_tail=log_tail(run_dir),
        )
    ref = ArtifactRef(type="QuantizedModel", path=str(out / "quantized"), hash=f"{run_id}:{model.hash}:{params.format}")
    warnings: tuple[tuple[str, Mapping[str, object]], ...] = ()
    warn = _parity_params(job.node_id, parity, "warn") if parity and parity.get("verdict") == "warn" else None
    if warn is not None:  # usable, but below the usual FP8 level: say so
        warnings = (("QNT-002", warn),)
    return NodeResult(outputs={"quantized": ref}, warnings=warnings)  # parity + sizes: <run_dir>/ptq/report.json


class QuantParityFailError(VlmPipelineError):
    """The quantized model disagrees with the BF16 one beyond the fail limits (ptq exit 10): not handed over."""

    code: ClassVar[str] = "QNT-001"


def _parity(report: Path) -> dict[str, Any] | None:
    """report["parity"] of the job (written by the job: read no-follow, bounded), or None."""
    try:
        data = read_small_json(report)
    except (OSError, ValueError):
        return None
    parity = data.get("parity") if isinstance(data, dict) else None
    return parity if isinstance(parity, dict) else None


def _num(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float("nan")


def _parity_params(node_id: str, parity: Mapping[str, Any], level: str) -> dict[str, object] | None:
    """QNT-001 / QNT-002 params; the limits are the report's own (``thresholds``), never a copy of ptq's constants.

    None when a number is missing: no text with "nan" (and no NaN in a JSON event); the caller falls back.
    """
    raw_limits = parity.get("thresholds")
    limits: Mapping[str, Any] = raw_limits if isinstance(raw_limits, dict) else {}
    top1, kl = _num(parity.get("top1_agreement_mean")), _num(parity.get("kl_mean"))
    limit_top1, limit_kl = _num(limits.get(f"{level}_top1")), _num(limits.get(f"{level}_kl"))
    samples = _num(parity.get("samples"))
    if not all(math.isfinite(v) for v in (top1, kl, limit_top1, limit_kl, samples)):
        return None
    kl_only = top1 >= limit_top1 and kl > limit_kl  # "top1" when both tripped
    return {
        "node": node_id,
        "top1_pct": round(top1 * 100, 1),
        "limit_top1_pct": round(limit_top1 * 100, 1),
        "kl": round(kl, 4),
        "limit_kl": round(limit_kl, 4),
        "sample_count": int(samples),
        "reason": "kl" if kl_only else "top1",
    }
