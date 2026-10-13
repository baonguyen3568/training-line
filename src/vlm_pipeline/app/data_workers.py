"""data.filter and data.dedupe on local datasets (Release 1.0 stream B). Host, CPU only: Pillow + hashlib.

Both keep the images where they are and write a new ``records.jsonl`` (a subset, records copied unchanged so a
caption's ``generated_by`` survives) plus a report of what was dropped and why. Images are opened only through
``common.image_rules`` and located with ``local_records.image_path`` (a link inside the folder may not point out).
v1 reads local datasets only; a hub dataset is refused at validate time (CFG-011, ``check_support``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_pipeline.common.dataset_ref import DatasetRef
from vlm_pipeline.common.image_rules import open_image, probe_image
from vlm_pipeline.common.local_records import Record, image_path, read_records, select, write_records
from vlm_pipeline.data.config import DedupeParams, FilterParams
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult, port_inputs

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker

FILTER_REPORT = "filter_report.json"
DEDUPE_REPORT = "dedupe_report.json"
_DHASH_BITS = 64


def check_support(node_id: str, params: object, model: object, inputs: Mapping[str, str]) -> None:
    """Pure rule (validate and run): every wired dataset is a local one (v1: no hub parquet read on the host)."""
    from vlm_pipeline.app.workers import NodeNotRunnableError

    if any(kind != "dataset:local" for kind in inputs.values()):
        raise NodeNotRunnableError("curation reads local datasets only (v1)", params={"node": node_id})


# ------------------------------------------------------------------------------------------------------- shared
def _records(worker: RealWorker, job: NodeJob, ref: ArtifactRef) -> tuple[DatasetRef, list[Record], Path]:
    from vlm_pipeline.app.workers import LocalDatasetInvalidError, NodeNotRunnableError

    src = worker.dataset_input(ref)
    if not src.records:
        raise NodeNotRunnableError("curation reads local datasets only (v1)", params={"node": job.node_id})
    try:  # records.jsonl may be tampered with: Record refuses ../ and absolute names (bao-security R1)
        records = select(read_records(worker.host_path(src.records)), part=src.split, seed=src.seed or 0,
                         val_fraction=src.val_fraction or 0.0, limit=src.limit)  # fmt: skip
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        bad = {"node": job.node_id, "problem_count": 1, "first_line": 0, "first_reason": "bad_image_name"}
        raise LocalDatasetInvalidError(f"unusable record: {type(exc).__name__}", params=bad) from None
    return src, records, worker.host_path(src.path)


def _image(folder: Path, record: Record) -> Path | None:
    try:
        return image_path(folder, record)
    except ValueError:
        return None


def _publish(
    worker: RealWorker, job: NodeJob, src: DatasetRef, kept: list[Record], *, report: dict[str, Any], name: str
) -> NodeResult:
    from vlm_pipeline.app.workers import LocalDatasetInvalidError

    run_dir = job.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if not kept:
        empty = {"node": job.node_id, "problem_count": report["input"], "first_line": 0, "first_reason": "no_samples"}
        raise LocalDatasetInvalidError("no sample left", params=empty)
    sha = write_records(run_dir / "records.jsonl", kept)
    dataset = DatasetRef(path=src.path, split="all", records=worker.container_path(run_dir / "records.jsonl"),
                         records_sha256=sha, seed=src.seed)  # fmt: skip
    return NodeResult(outputs={"dataset": worker.dataset_artifact(run_dir / "dataset.json", dataset)})


def _target_chars(target: Any) -> int:
    return len(target) if isinstance(target, str) else len(json.dumps(target, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------------- filter
def _target_reason(record: Record, p: FilterParams) -> str | None:
    chars = _target_chars(record.target)
    if chars < p.min_target_chars:
        return "target_too_short"
    return "target_too_long" if chars > p.max_target_chars else None


def _image_reason(folder: Path, record: Record, p: FilterParams) -> str | None:
    path = _image(folder, record)
    if path is None:
        return "image_outside_folder"
    try:
        width, height, _fmt = probe_image(path)
    except (OSError, ValueError):  # missing, a link, too large, unknown format, decompression bomb
        return "image_broken"
    limits = (
        (min(width, height) < p.min_short_side_px, "image_too_small"),
        (max(width, height) > p.max_long_side_px, "image_too_large"),
        (width * height > p.max_megapixels * 1_000_000, "image_too_many_pixels"),
    )
    return next((reason for broken, reason in limits if broken), None)


def _drop_reason(folder: Path, record: Record, p: FilterParams) -> str | None:
    return _target_reason(record, p) or _image_reason(folder, record, p)


def run_filter(worker: RealWorker, job: NodeJob) -> NodeResult:
    """Drop samples whose image or answer falls outside the limits; a broken image fails the node unless dropped."""
    from vlm_pipeline.app.workers import LocalDatasetInvalidError

    params = job.params
    assert isinstance(params, FilterParams)  # noqa: S101 - registry guarantees the params model
    src, records, folder = _records(worker, job, job.inputs["dataset"])
    kept: list[Record] = []
    dropped: Counter[str] = Counter()
    for n, record in enumerate(records, start=1):
        reason = _drop_reason(folder, record, params)
        if reason == "image_broken" and not params.drop_broken:
            bad = {"node": job.node_id, "problem_count": 1, "first_line": n, "first_reason": "image_unreadable"}
            raise LocalDatasetInvalidError("broken image (drop_broken is off)", params=bad)
        if reason is None:
            kept.append(record)
        else:
            dropped[reason] += 1
    report = {"input": len(records), "kept": len(kept), "dropped": dict(sorted(dropped.items())),
              "limits": params.model_dump()}  # fmt: skip
    return _publish(worker, job, src, kept, report=report, name=FILTER_REPORT)


# ------------------------------------------------------------------------------------------------------- dedupe
def dhash(path: Path) -> int:
    """64-bit difference hash: 9x8 grayscale, each bit = left pixel brighter than its right neighbour."""
    from PIL import Image

    img = open_image(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = img.tobytes()  # mode L: one byte per pixel, row major
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


class _NearIndex:
    """dhash values within ``max_distance`` bits, found by pigeonhole: of ``max_distance + 1`` bands, one matches.

    Fast at the default distance (4: five ~13-bit bands). At the maximum (16: 3-4 bit bands) buckets hold ~n/10
    hashes, so it nears O(n^2 / 10): fine up to tens of thousands of images, slow beyond 100k.
    """

    def __init__(self, max_distance: int) -> None:
        self.d = max_distance
        n = max_distance + 1
        self.bands = [(i * _DHASH_BITS // n, (i + 1) * _DHASH_BITS // n) for i in range(n)]
        self.buckets: dict[tuple[int, int], list[int]] = {}

    def _keys(self, h: int) -> Iterable[tuple[int, int]]:
        for i, (lo, hi) in enumerate(self.bands):
            yield i, (h >> lo) & ((1 << (hi - lo)) - 1)

    def near(self, h: int) -> bool:
        seen: set[int] = set()
        for key in self._keys(h):
            for other in self.buckets.get(key, ()):
                if other in seen:
                    continue
                seen.add(other)
                if bin(h ^ other).count("1") <= self.d:
                    return True
        return False

    def add(self, h: int) -> None:
        for key in self._keys(h):
            self.buckets.setdefault(key, []).append(h)


def _key(folder: Path, record: Record, compare: str) -> int | str | None:
    if compare == "sha256":
        return record.sha256
    path = _image(folder, record)
    if path is None:
        return None
    try:
        return dhash(path)
    except (OSError, ValueError):
        return None


def run_dedupe(worker: RealWorker, job: NodeJob) -> NodeResult:
    """Keep the first of identical (sha256) or near-identical (dhash) images.

    With check_leak, also drop every sample whose image is in a held-out set wired to ``heldout`` (all of them: a
    many port).
    """
    params = job.params
    assert isinstance(params, DedupeParams)  # noqa: S101 - registry guarantees the params model
    src, records, folder = _records(worker, job, job.inputs["dataset"])
    exact: set[int | str] = set()
    near = _NearIndex(params.dhash_max_distance)
    heldout_exact: set[int | str] = set()
    heldout_near = _NearIndex(params.dhash_max_distance)
    heldout_sets = port_inputs(job.inputs, "heldout") if params.check_leak else []
    for ref in heldout_sets:
        _, held, held_folder = _records(worker, job, ref)
        for record in held:
            key = _key(held_folder, record, params.compare)
            if isinstance(key, int):
                heldout_near.add(key)
            if key is not None:
                heldout_exact.add(key)
    kept: list[Record] = []
    dropped: Counter[str] = Counter()
    for record in records:
        key = _key(folder, record, params.compare)
        if key is None:
            dropped["image_unreadable"] += 1
            continue
        if key in heldout_exact or (isinstance(key, int) and heldout_near.near(key)):
            dropped["in_heldout"] += 1
            continue
        if key in exact or (isinstance(key, int) and near.near(key)):
            dropped["duplicate"] += 1
            continue
        exact.add(key)
        if isinstance(key, int):
            near.add(key)
        kept.append(record)
    report = {"input": len(records), "kept": len(kept), "dropped": dict(sorted(dropped.items())),
              "compare": params.compare, "dhash_max_distance": params.dhash_max_distance,
              "heldout_sets": len(heldout_sets)}  # fmt: skip
    return _publish(worker, job, src, kept, report=report, name=DEDUPE_REPORT)


__all__ = ["DEDUPE_REPORT", "FILTER_REPORT", "check_support", "dhash", "run_dedupe", "run_filter"]
