"""Node ``quantize.qat`` for the real worker (ADR-022 #10): host side of ``vlm_pipeline.quantize.qat``.

Wired into ``RealWorker._handlers`` as ``functools.partial(run_qat, self)``. The host chooses the samples (local
dataset, its ``train`` part, limit) and writes them to ``qat_data.jsonl`` in the run dir (image names relative to the
dataset folder, untrusted text as data); the container trains on that file only. The job is narrowed: model and
dataset folder read-only, writes only its run dir. Calibration uses the first training images (v1).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from vlm_pipeline.common.errors import InsufficientMemoryError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.local_records import DEFAULT_PROMPT, image_path, read_records, select, target_text
from vlm_pipeline.common.run_log import log_tail
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.quantize.config import QatParams
from vlm_pipeline.quantize.qat import check_support

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker

DATA_NAME = "qat_data.jsonl"


def est_peak_gb(disk_gb: float, adapter: str) -> float:
    """BF16 weights twice (model + calibration / export) + fp32 Adam on the trainable part + activations and JIT.

    Placeholder until the Q2 smoke measures it (fixcode): LoRA ~ PTQ + 4 GB; full adds ~8 bytes/param of Adam state.
    """
    full_state = disk_gb * 4 if adapter == "full" else 0.0
    return round(disk_gb * 2 + 20 + full_state, 1)


def run_qat(worker: RealWorker, job: NodeJob) -> NodeResult:
    from vlm_pipeline.app.workers import LocalDatasetInvalidError

    params = job.params
    assert isinstance(params, QatParams)  # noqa: S101 - registry guarantees the params model
    model = job.inputs["model"]
    entry = worker.base_of(model)
    data = worker.dataset_input(job.inputs["train_data"])
    kinds = {"train_data": "dataset:local" if data.records else "dataset:hub"}
    if "calib_data" in job.inputs:
        kinds["calib_data"] = "dataset:local"
    check_support(job.node_id, params, entry, kinds)
    assert data.records is not None  # noqa: S101 - check_support refuses hub data
    bad = {"node": job.node_id, "problem_count": 1, "first_line": 0, "first_reason": "bad_image_name"}
    folder = worker.host_path(data.path)
    try:  # records.jsonl may be tampered with: Record rejects ../ and absolute names (security R1)
        records = select(read_records(worker.host_path(data.records)), part=data.split, seed=data.seed or 0,
                         val_fraction=data.val_fraction or 0.0, limit=data.limit)  # fmt: skip
        for record in records:
            image_path(folder, record)  # a link inside the folder may not point out
    except ValueError:  # pydantic ValidationError is a ValueError
        raise LocalDatasetInvalidError("train image outside its folder", params=bad) from None
    lines = []
    for record in records:
        answer = target_text(record.target, "chat")
        if answer.strip():
            lines.append({"image": record.image, "prompt": record.prompt or DEFAULT_PROMPT, "answer": answer})
    if not lines:
        empty = {"node": job.node_id, "problem_count": 0, "first_line": 0, "first_reason": "no_samples"}
        raise LocalDatasetInvalidError("no training sample for QAT", params=empty)
    run_dir = job.run_dir.resolve()
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    data_file = run_dir / DATA_NAME
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    fd = os.open(data_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)  # never through a link
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    out = run_dir / "qat"
    run_id = run_dir.name[:120]
    manifest = RunManifest(
        run_id=run_id, name=f"{job.spec.type_id} {entry.display_name}", stage="quantize", task=f"qat-{params.format}",
        base_model=entry.repo, base_model_revision=entry.revision, dataset=data.hash, container_image=worker.image,
    )  # fmt: skip
    argv = (
        "python", "-m", "vlm_pipeline.quantize.qat",
        "--model", worker.container_path(worker.model_dir(model)),
        "--data", worker.container_path(data_file),
        "--images", data.path,
        "--out", worker.container_path(out),
        "--format", params.format, "--adapter", params.adapter, "--epochs", str(params.epochs),
        "--lr", repr(params.learning_rate), "--max-steps", str(params.max_steps),
        "--calib-size", str(params.calib_size),
    )  # fmt: skip
    container = ContainerJob(
        name=run_id, image=worker.image, argv=argv, read_only_project=True, rw_dirs=(run_dir,), narrow=True,
        ro_inputs=(worker.mount_input(worker.model_dir(model)), worker.mount_input(data.path)),
    )  # fmt: skip
    need = est_peak_gb(entry.disk_gb, params.adapter)
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
            log_tail=log_tail(run_dir),
        )
    ref = ArtifactRef(
        type="QuantizedModel", path=str(out / "quantized"), hash=f"{run_id}:{model.hash}:qat-{params.format}"
    )
    return NodeResult(outputs={"quantized": ref})  # loss curve, merge drift, sizes: <run_dir>/qat/report.json


__all__ = ["DATA_NAME", "est_peak_gb", "run_qat"]
