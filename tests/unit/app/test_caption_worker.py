"""data.caption_vlm (Release 1.0 luồng B): one vLLM batch job captions a local dataset; untrusted text is bounded."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import caption_worker, trial_worker
from vlm_pipeline.app.caption_worker import MAX_CAPTION_CHARS, check_support, run_caption
from vlm_pipeline.app.workers import LocalDatasetInvalidError, NodeNotRunnableError, RealWorker
from vlm_pipeline.common.dataset_ref import DatasetRef, write_dataset_ref
from vlm_pipeline.common.local_records import Record, read_records, write_records
from vlm_pipeline.common.runinfo import RunStatus
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY

MODEL = "cosmos-reason2-2b"


def _root(tmp_path: Path) -> Path:
    from vlm_pipeline.common.paths import project_root

    root = tmp_path / "proj"
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")
    return root


def _model(w: RealWorker) -> ArtifactRef:
    entry = w.catalog.get(MODEL)
    assert entry is not None
    snap = w.root / ".cache" / "hf" / "hub" / f"models--{entry.repo.replace('/', '--')}" / "snapshots" / entry.revision
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    spec = REGISTRY.get("model.hf_pick")
    assert spec is not None
    pick = NodeJob("m", spec, spec.validate_params({"model": MODEL}), {}, w.root / "outputs" / "graph" / "runs" / "m")
    return w.run(pick).outputs["model"]


def _dataset(w: RealWorker, targets: list[Any]) -> ArtifactRef:
    folder = w.root / "data" / "mine"
    folder.mkdir(parents=True)
    recs = []
    for n, target in enumerate(targets):
        (folder / f"{n}.png").write_bytes(b"x")
        recs.append(Record(image=f"{n}.png", target=target, sha256=hashlib.sha256(str(n).encode()).hexdigest()))
    run = w.root / "outputs" / "graph" / "runs" / "d"
    sha = write_records(run / "records.jsonl", recs)
    ref = DatasetRef(path=w.container_path(folder), split="all", records=w.container_path(run / "records.jsonl"),
                     records_sha256=sha, seed=0)  # fmt: skip
    write_dataset_ref(run / "dataset.json", ref)
    return ArtifactRef(type="Dataset", path=str(run / "dataset.json"), hash=ref.hash)


def _job(w: RealWorker, params: dict[str, Any], dataset: ArtifactRef, model: ArtifactRef) -> NodeJob:
    spec = REGISTRY.get("data.caption_vlm")
    assert spec is not None
    run = w.root / "outputs" / "graph" / "runs" / "cap"
    return NodeJob("cap", spec, spec.validate_params(params), {"dataset": dataset, "model": model}, run)


def _fake_batch(answers: dict[str, dict[str, Any]], seen: dict[str, Any]) -> Any:
    from vlm_pipeline.serve.vllm_runner import BatchResult

    def fake(root: Path, model: Path, items: list[dict[str, Any]], **kw: Any) -> Any:
        seen.update(items=items, kw=kw)
        run_dir = kw["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"id": i["id"], **answers[i["id"]]}) for i in items]
        (run_dir / "out.jsonl").write_text("\n".join(lines) + "\n")
        (run_dir / "engine.json").write_text(json.dumps({"vllm": "0.31.0"}))
        return BatchResult(run_dir=run_dir, status=RunStatus.COMPLETED, error_code=None, container_digest="sha256:b")

    return fake


@pytest.fixture
def parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trial_worker, "VLLM_PARITY_MODELS", frozenset({MODEL}))


@pytest.mark.usefixtures("parity")
def test_only_missing_targets_are_captioned_and_failures_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vlm_pipeline.serve import vllm_runner

    w = RealWorker(_root(tmp_path))
    model, data = _model(w), _dataset(w, ["keep me", "", None])
    seen: dict[str, Any] = {}
    answers = {"1": {"text": "  a cat <script>x</script>  " + "z" * 9000}, "2": {"error": {"reason": "unreadable"}}}
    monkeypatch.setattr(vllm_runner, "run_vllm_batch", _fake_batch(answers, seen))
    out = run_caption(w, _job(w, {"prompt": "Describe."}, data, model)).outputs["dataset"]
    assert [i["id"] for i in seen["items"]] == ["1", "2"]  # record 0 already has a target
    assert all(i["prompt"] == "Describe." for i in seen["items"])
    run = w.root / "outputs" / "graph" / "runs" / "cap"
    records = list(read_records(run / "records.jsonl"))
    assert [r.image for r in records] == ["0.png", "1.png"]  # the unreadable image is dropped
    assert records[0].target == "keep me"
    assert records[1].target.startswith("a cat <script>")  # stored as data, bounded; never rendered as markup
    assert len(records[1].target) == MAX_CAPTION_CHARS
    report = json.loads((run / "caption_report.json").read_text())
    assert (report["captioned"], report["failed_images"], report["kept_existing"]) == (1, 1, 1)
    assert out.type == "Dataset"


@pytest.mark.usefixtures("parity")
def test_no_sample_left_is_data006(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.serve import vllm_runner

    w = RealWorker(_root(tmp_path))
    model, data = _model(w), _dataset(w, [""])
    monkeypatch.setattr(vllm_runner, "run_vllm_batch", _fake_batch({"0": {"error": {"reason": "unreadable"}}}, {}))
    with pytest.raises(LocalDatasetInvalidError) as ei:
        run_caption(w, _job(w, {}, data, model))
    assert ei.value.params["first_reason"] == "no_samples"


def test_models_without_vllm_parity_are_refused_before_any_job(tmp_path: Path) -> None:
    w = RealWorker(_root(tmp_path))
    entry = w.catalog.get(MODEL)
    assert entry is not None
    with pytest.raises(NodeNotRunnableError):
        check_support("cap", None, entry, {"dataset": "dataset:local"})
    assert caption_worker.MAX_CAPTION_CHARS == 8000


@pytest.mark.parametrize("name", ["../x.png", "/etc/x.png", "a/../../x.png", "./a.png", "a//b.png", "a\\b.png", "a\nb"])
def test_record_image_must_stay_inside_the_dataset_folder(name: str) -> None:
    """bao-security R1: a tampered records.jsonl cannot make a job read (and caption) another run's or user's image."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="relative path inside"):
        Record(image=name, target="t", sha256="0" * 64)


def test_a_link_inside_the_folder_cannot_point_out(tmp_path: Path) -> None:
    from vlm_pipeline.common.local_records import image_path

    folder = tmp_path / "d"
    folder.mkdir()
    (tmp_path / "other.png").write_bytes(b"x")
    (folder / "in.png").symlink_to(tmp_path / "other.png")
    with pytest.raises(ValueError, match="outside the dataset folder"):
        image_path(folder, Record(image="in.png", target="t", sha256="0" * 64))


def test_generated_by_is_written_only_when_set(tmp_path: Path) -> None:
    """User-labelled records keep their bytes (and sha256, i.e. cache keys); captions carry their model."""
    plain = Record(image="a.png", target="t", sha256="0" * 64)
    made = plain.model_copy(update={"generated_by": "m@r"})
    write_records(tmp_path / "a.jsonl", [plain])
    write_records(tmp_path / "b.jsonl", [made])
    assert "generated_by" not in (tmp_path / "a.jsonl").read_text()
    assert [r.generated_by for r in read_records(tmp_path / "b.jsonl")] == ["m@r"]


@pytest.mark.usefixtures("parity")
def test_captions_carry_their_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.serve import vllm_runner

    w = RealWorker(_root(tmp_path))
    model, data = _model(w), _dataset(w, ["", "mine"])
    monkeypatch.setattr(vllm_runner, "run_vllm_batch", _fake_batch({"0": {"text": "cap"}}, {}))
    run_caption(w, _job(w, {}, data, model))
    recs = list(read_records(w.root / "outputs" / "graph" / "runs" / "cap" / "records.jsonl"))
    entry = w.catalog.get(MODEL)
    assert entry is not None
    assert [r.generated_by for r in recs] == [f"{MODEL}@{entry.revision}", None]
