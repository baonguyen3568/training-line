"""Host side of ``data.caption_vlm`` (Release 1.0 luồng B): a catalog VLM writes the target text of each image.

One offline vLLM job for the whole dataset (``serve.vllm_runner.run_vllm_batch``: GPU lock, memory queue, memguard,
``--network none``, partial results per chunk), greedy, the model's training pixels. v1 takes only models whose vLLM
answers passed the vLLM<->HF parity gate (``trial_worker.VLLM_PARITY_MODELS``); others are refused at validate time
(CFG-011) rather than answered by a second, slower engine.

Input: a local dataset (``records.jsonl``, common.local_records). Output: the same images with new records:
- a record is captioned when it has no target, or always with ``overwrite_existing``; the others pass through;
- a caption is UNTRUSTED text (bao-security §B3): stripped, cut at ``MAX_CAPTION_CHARS``, written by the JSON
  serializer only; the prompt reaches the job through the batch file in the run dir, never argv;
- an image the container refuses (image rules) drops that record and is counted; no record left -> DATA-006;
- ``caption_report.json`` records counts, model + revision (its licence carries over), prompt sha and the engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.dataset_ref import DatasetRef
from vlm_pipeline.common.errors import InsufficientMemoryError
from vlm_pipeline.common.jobrun import MemoryPreemptedBeforeStartError
from vlm_pipeline.common.local_records import Record, image_path, read_records, select, write_records
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.data.config import CaptionParams
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import NodeJob, NodeResult

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker

MAX_CAPTION_CHARS = 8000
REPORT_NAME = "caption_report.json"


def check_support(node_id: str, params: object, model: CatalogEntry | None, inputs: Mapping[str, str]) -> None:
    """Pure rule, at validate time and before the job: native model with vLLM parity, local dataset (v1)."""
    from vlm_pipeline.app.trial_worker import VLLM_PARITY_MODELS
    from vlm_pipeline.app.workers import NodeNotRunnableError

    refuse = {"node": node_id, "node_type": "data.caption_vlm"}
    if getattr(params, "task", "caption") == "qa":  # ADR-022 #6: the qa worker path is not built yet
        raise NodeNotRunnableError("caption task qa is not runnable yet", params={**refuse, "reason": "qa_pending"})
    if model is not None and (model.auto_class != "AutoModelForImageTextToText" or model.id not in VLLM_PARITY_MODELS):
        raise NodeNotRunnableError("caption needs a native model that passed the vLLM parity gate", params=refuse)
    if inputs.get("dataset", "dataset:local") != "dataset:local":  # allowlist; unwired is CFG-003's job
        raise NodeNotRunnableError("caption reads local datasets only (v1)", params=refuse)


def _needs_caption(record: Record, overwrite: bool) -> bool:
    return overwrite or record.target is None or (isinstance(record.target, str) and not record.target.strip())


def clean_caption(text: str) -> str:
    """Untrusted model text as stored: stripped and bounded (never rendered as markup, see CaptionParams)."""
    return text.strip()[:MAX_CAPTION_CHARS]


def _caption_batch(
    worker: RealWorker, job: NodeJob, requests: list[dict[str, Any]], batch: Mapping[str, Any]
) -> tuple[dict[int, str], set[int], dict[str, Any]]:
    """One vLLM job for every request: (captions by record index, refused images, engine facts)."""
    from vlm_pipeline.serve.vllm_runner import run_vllm_batch

    run_dir = job.run_dir.resolve()
    try:
        res = run_vllm_batch(
            worker.root, batch["model_dir"], requests, who=worker.who, min_pixels=batch["min_pixels"],
            max_pixels=batch["max_pixels"], cancel=job.cancel, on_wait=job.on_wait, run_dir=run_dir,
            manifest=batch["manifest"],
        )  # fmt: skip
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc)) from exc
    try:
        rows = res.rows() if res.ok else []
        engine = res.engine() if res.ok else {}
    except (OSError, ValueError):  # a job-made link / FIFO / garbage: no answers
        rows, engine = [], {}
    if not rows:
        from vlm_pipeline.common.run_log import log_tail

        raise ContainerExit(
            75 if res.status == RunStatus.INTERRUPTED else 1,
            cancelled_by_executor=res.status == RunStatus.CANCELLED,
            log_tail=log_tail(run_dir),
        )
    captions = {int(r["id"]): clean_caption(str(r["text"])) for r in rows if "text" in r}
    failed = {int(r["id"]) for r in rows if "text" not in r}
    return captions, failed, engine


def run_caption(worker: RealWorker, job: NodeJob) -> NodeResult:
    from vlm_pipeline.app.trial_worker import _pixels, base_of
    from vlm_pipeline.app.workers import LocalDatasetInvalidError, NodeNotRunnableError

    params = job.params
    assert isinstance(params, CaptionParams)  # noqa: S101 - registry guarantees the params model
    src = worker.dataset_input(job.inputs["dataset"])
    if not src.records:
        refuse = {"node": job.node_id, "node_type": job.spec.type_id}
        raise NodeNotRunnableError("caption reads local datasets only (v1)", params=refuse)
    ref = job.inputs["model"]
    entry, _ = base_of(worker, ref)
    check_support(job.node_id, params, entry, {"dataset": "dataset:local"})
    try:  # records.jsonl may come from a tampered file: Record rejects ../ and absolute names (bao-security R1)
        records = select(read_records(worker.host_path(src.records)), part=src.split, seed=src.seed or 0,
                         val_fraction=src.val_fraction or 0.0, limit=src.limit)  # fmt: skip
        folder = worker.host_path(src.path)
        todo = [n for n, r in enumerate(records) if _needs_caption(r, params.overwrite_existing)]
        images = {n: image_path(folder, records[n]) for n in todo}  # a link inside the folder may not point out
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        bad = {"node": job.node_id, "problem_count": 1, "first_line": 0, "first_reason": "bad_image_name"}
        raise LocalDatasetInvalidError(f"unusable record: {type(exc).__name__}", params=bad) from None
    min_px, max_px = _pixels(worker, entry, ref)
    run_dir = job.run_dir.resolve()
    manifest = RunManifest(run_id=run_dir.name[:120], name=f"{job.spec.type_id} {entry.display_name}", stage="data",
                           task="caption", base_model=entry.repo, base_model_revision=entry.revision)  # fmt: skip
    requests = [{"id": str(n), "image": str(images[n]), "prompt": params.prompt,
                 "max_tokens": params.max_new_tokens} for n in todo]  # fmt: skip
    batch = {"model_dir": worker.model_dir(ref), "min_pixels": min_px, "max_pixels": max_px, "manifest": manifest}
    captions, failed, engine = _caption_batch(worker, job, requests, batch) if todo else ({}, set(), {})
    made_by = f"{entry.id}@{entry.revision}"
    out = [r.model_copy(update={"target": captions[n], "generated_by": made_by}) if n in captions else r
           for n, r in enumerate(records) if n not in failed]  # fmt: skip
    if not out:
        empty = {"node": job.node_id, "problem_count": len(failed), "first_line": 0, "first_reason": "no_samples"}
        raise LocalDatasetInvalidError("no sample left after captioning", params=empty)
    sha = write_records(run_dir / "records.jsonl", out)
    report = {
        "captioned": len(captions),
        "failed_images": len(failed),
        "kept_existing": len(records) - len(todo),
        "model": entry.repo,
        "revision": entry.revision,
        "license": entry.license,
        "prompt_sha256": hashlib.sha256(params.prompt.encode()).hexdigest(),
        "max_new_tokens": params.max_new_tokens,
        "engine": {k: engine.get(k) for k in ("vllm", "quantization", "dtype", "container_digest")},
    }
    (run_dir / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    dataset = DatasetRef(path=src.path, split="all", records=worker.container_path(run_dir / "records.jsonl"),
                         records_sha256=sha, seed=src.seed)  # fmt: skip
    return NodeResult(outputs={"dataset": worker.dataset_artifact(run_dir / "dataset.json", dataset)})


__all__ = ["MAX_CAPTION_CHARS", "REPORT_NAME", "check_support", "clean_caption", "run_caption"]
