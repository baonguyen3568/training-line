"""Node ``quantize.modelopt_ptq`` for the real worker (B-4): FP8 PTQ of a native VLM in the training image.

Wired into ``RealWorker._handlers`` as ``functools.partial(run_quantize, self)`` (same pattern as the trial worker).
The container runs ``vlm_pipeline.quantize.ptq``; the job writes only its own run dir (project read-only).
Scope, memory estimate and argv: ``vlm_pipeline.quantize.scope``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_pipeline.common.errors import InsufficientMemoryError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
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
        read_only_project=True,  # writes only its run dir; the FP8 extension is compiled into the container's /tmp
        rw_dirs=(run_dir,),
        env={"TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions", "MAX_JOBS": "2"},  # noqa: S108
    )
    need = est_peak_gb(entry)
    try:
        final = run_job(
            container, manifest, run_dir, project_root=worker.root, who=worker.who, est_peak_gb=need,
            ckpt_gb=entry.disk_gb, cancel=job.cancel, require_approved_image=True, on_wait=job.on_wait,
        )  # fmt: skip
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc), params={"need_gb": need}) from exc
    if final.status != RunStatus.COMPLETED:
        raise ContainerExit(
            final.exit_code or 1,
            oom_killed=bool(final.oom_killed),
            cancelled_by_executor=final.status == RunStatus.CANCELLED,
            log_tail=_tail(run_dir / "logs" / "stdout.log"),
        )
    ref = ArtifactRef(type="QuantizedModel", path=str(out / "quantized"), hash=f"{run_id}:{model.hash}:{params.format}")
    return NodeResult(outputs={"quantized": ref})  # parity + sizes: <run_dir>/ptq/report.json


def _tail(path: Any, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""
