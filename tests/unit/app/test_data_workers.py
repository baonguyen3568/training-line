"""data.filter / data.dedupe on local datasets (Release 1.0 stream B): subsets of records, reports, no image moved."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import data_workers
from vlm_pipeline.app.workers import LocalDatasetInvalidError, RealWorker
from vlm_pipeline.common.dataset_ref import read_dataset_ref
from vlm_pipeline.common.local_records import read_records
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY

pil = pytest.importorskip("PIL.Image")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    return root


def _folder(root: Path, name: str, images: dict[str, Any]) -> Path:
    """images: file name -> (w, h, color) or raw bytes; target = the name, labels.jsonl in order."""
    folder = root / "data" / name
    folder.mkdir(parents=True)
    lines = []
    for fname, spec in images.items():
        if isinstance(spec, bytes):
            (folder / fname).write_bytes(spec)
        else:
            w, h, color = spec
            if color == "pattern":
                img = pil.linear_gradient("L").resize((w, h)).convert("RGB")
            else:
                img = pil.new("RGB", (w, h), color)
            img.save(folder / fname)
        lines.append(json.dumps({"image": fname, "target": f"answer for {fname}"}))
    (folder / "labels.jsonl").write_text("\n".join(lines) + "\n")
    return folder


def _job(type_id: str, params: dict[str, Any], run_dir: Path, inputs: dict[str, ArtifactRef]) -> NodeJob:
    spec = REGISTRY.get(type_id)
    assert spec is not None
    return NodeJob(run_dir.name, spec, spec.validate_params(params), inputs, run_dir)


def _load(w: RealWorker, name: str, **params: Any) -> ArtifactRef:
    run_dir = w.root / "outputs" / "graph" / "runs" / f"lf-{name}"
    return w.run(_job("data.local_folder", {"path": f"data/{name}", **params}, run_dir, {})).outputs["dataset"]


def _images(w: RealWorker, ref: ArtifactRef) -> list[str]:
    return [r.image for r in read_records(w.host_path(read_dataset_ref(Path(ref.path)).records or ""))]


def test_filter_drops_what_is_outside_the_limits_and_reports_why(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _folder(root, "mine", {"ok.png": (128, 96, "red"), "tiny.png": (20, 20, "red"), "wide.png": (5000, 600, "red")})
    w = RealWorker(root)
    run_dir = root / "outputs" / "graph" / "runs" / "flt"
    src = _load(w, "mine", max_image_side_px=8192)  # the folder scan's own cap is wider than the filter's
    out = w.run(_job("data.filter", {}, run_dir, {"dataset": src})).outputs["dataset"]
    assert _images(w, out) == ["ok.png"]
    report = json.loads((run_dir / data_workers.FILTER_REPORT).read_text())
    assert report["dropped"] == {"image_too_large": 1, "image_too_small": 1}
    ref = read_dataset_ref(Path(out.path))
    assert ref.path.endswith("/data/mine")  # images stay where they are
    assert ref.split == "all"


def test_a_broken_image_fails_the_filter_unless_it_drops_broken_ones(tmp_path: Path) -> None:
    root = _root(tmp_path)
    folder = _folder(root, "mine", {"ok.png": (128, 96, "red"), "bad.png": (128, 96, "red")})
    w = RealWorker(root)
    src = _load(w, "mine")
    (folder / "bad.png").write_bytes(b"not an image")  # broken after the folder was scanned
    with pytest.raises(LocalDatasetInvalidError) as err:
        w.run(_job("data.filter", {"drop_broken": False}, root / "outputs" / "graph" / "runs" / "f1", {"dataset": src}))
    assert err.value.params["first_reason"] == "image_unreadable"
    out = w.run(_job("data.filter", {}, root / "outputs" / "graph" / "runs" / "f2", {"dataset": src}))
    assert _images(w, out.outputs["dataset"]) == ["ok.png"]


def test_dedupe_keeps_the_first_copy_and_drops_heldout_images(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _folder(root, "mine", {"a.png": (64, 64, "red"), "b.png": (64, 64, "red"), "c.png": (64, 64, "blue"),
                           "d.png": (64, 64, "green")})  # fmt: skip
    _folder(root, "eval1", {"x.png": (64, 64, "blue")})
    _folder(root, "eval2", {"y.png": (64, 64, "green")})
    w = RealWorker(root)
    inputs = {"dataset": _load(w, "mine"), "heldout": _load(w, "eval1"), "heldout.1": _load(w, "eval2")}
    run_dir = root / "outputs" / "graph" / "runs" / "dd"
    out = w.run(_job("data.dedupe", {}, run_dir, inputs)).outputs["dataset"]
    assert _images(w, out) == ["a.png"]  # b = copy of a; c and d are in the two held-out sets
    report = json.loads((run_dir / data_workers.DEDUPE_REPORT).read_text())
    assert report["dropped"] == {"duplicate": 1, "in_heldout": 2}
    assert report["heldout_sets"] == 2


def test_dhash_finds_a_resized_copy_that_sha256_misses(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _folder(root, "mine", {"a.png": (256, 256, "pattern"), "a_small.png": (128, 128, "pattern")})
    w = RealWorker(root)
    src = _load(w, "mine")
    by_sha = w.run(_job("data.dedupe", {}, root / "outputs" / "graph" / "runs" / "d1", {"dataset": src}))
    assert len(_images(w, by_sha.outputs["dataset"])) == 2
    by_dhash = w.run(_job("data.dedupe", {"compare": "dhash"}, root / "outputs" / "graph" / "runs" / "d2",
                          {"dataset": src}))  # fmt: skip
    assert _images(w, by_dhash.outputs["dataset"]) == ["a.png"]


def test_near_index_matches_within_the_distance_only() -> None:
    idx = data_workers._NearIndex(4)
    idx.add(0)
    assert idx.near(0b1111)  # 4 bits apart
    assert not idx.near(0b11111)  # 5 bits apart


def test_a_generated_target_keeps_its_provenance_through_curation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _folder(root, "mine", {"a.png": (64, 64, "red")})
    w = RealWorker(root)
    src = _load(w, "mine")
    records_path = w.host_path(read_dataset_ref(Path(src.path)).records or "")
    [record] = list(read_records(records_path))
    from vlm_pipeline.common.local_records import write_records

    write_records(records_path, [record.model_copy(update={"generated_by": "qwen@abc"})])
    out = w.run(_job("data.filter", {}, root / "outputs" / "graph" / "runs" / "f", {"dataset": src}))
    path = w.host_path(read_dataset_ref(Path(out.outputs["dataset"].path)).records or "")
    assert [r.generated_by for r in read_records(path)] == ["qwen@abc"]


def test_curation_refuses_hub_data() -> None:
    from vlm_pipeline.app.workers import NodeNotRunnableError

    with pytest.raises(NodeNotRunnableError):
        data_workers.check_support("n", None, None, {"dataset": "dataset:hub"})
    data_workers.check_support("n", None, None, {"dataset": "dataset:local", "heldout": "dataset:local"})
