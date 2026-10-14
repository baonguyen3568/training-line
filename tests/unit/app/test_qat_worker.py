"""quantize.qat host side (ADR-022 #10): the samples file, the narrowed job, the refusals before any job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import qat_worker
from vlm_pipeline.app.qat_worker import DATA_NAME, est_peak_gb, run_qat
from vlm_pipeline.app.workers import RealWorker
from vlm_pipeline.common.dataset_ref import DatasetRef, write_dataset_ref
from vlm_pipeline.common.local_records import Record, write_records
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.quantize import qat
from vlm_pipeline.quantize.scope import QuantizeNotSupportedError

MODEL = "cosmos-reason2-2b"  # quant_fp8: official in the catalog


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


def _dataset(w: RealWorker, targets: list[Any], prompts: list[str | None]) -> ArtifactRef:
    folder = w.root / "data" / "mine"
    folder.mkdir(parents=True)
    recs = []
    for n, (target, prompt) in enumerate(zip(targets, prompts, strict=True)):
        (folder / f"{n}.png").write_bytes(b"x")
        sha = hashlib.sha256(str(n).encode()).hexdigest()
        recs.append(Record(image=f"{n}.png", target=target, prompt=prompt, sha256=sha))
    run = w.root / "outputs" / "graph" / "runs" / "d"
    sha = write_records(run / "records.jsonl", recs)
    ref = DatasetRef(path=w.container_path(folder), split="all", records=w.container_path(run / "records.jsonl"),
                     records_sha256=sha, seed=0)  # fmt: skip
    write_dataset_ref(run / "dataset.json", ref)
    return ArtifactRef(type="Dataset", path=str(run / "dataset.json"), hash=ref.hash)


def _job(w: RealWorker, params: dict[str, Any], inputs: dict[str, ArtifactRef]) -> NodeJob:
    spec = REGISTRY.get("quantize.qat")
    assert spec is not None
    return NodeJob("q", spec, spec.validate_params(params), inputs, w.root / "outputs" / "graph" / "runs" / "q")


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qat, "WORKER_READY", True)


def test_refused_until_the_gpu_smoke_passes(tmp_path: Path) -> None:
    w = RealWorker(_root(tmp_path))
    job = _job(w, {}, {"model": _model(w), "train_data": _dataset(w, ["a"], [None])})
    with pytest.raises(QuantizeNotSupportedError) as err:
        run_qat(w, job)
    assert err.value.params["reason"] == "worker_pending"
    assert not job.run_dir.exists()


@pytest.mark.usefixtures("ready")
def test_samples_file_and_narrowed_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    model = _model(w)
    data = _dataset(w, [{"total": "9"}, "", "a cat"], [None, None, "What is it?"])
    seen: dict[str, Any] = {}

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.update(job=job, kw=kw)
        return manifest.model_copy(update={"status": RunStatus.COMPLETED})

    monkeypatch.setattr(qat_worker, "run_job", fake_run_job)
    out = run_qat(
        w, _job(w, {"adapter": "lora", "lr_preset": "low", "max_steps": 20}, {"model": model, "train_data": data})
    )
    run = w.root / "outputs" / "graph" / "runs" / "q"
    rows = [json.loads(line) for line in (run / DATA_NAME).read_text().splitlines()]
    assert rows == [  # the empty target is left out; a missing prompt gets the training default
        {"image": "0.png", "prompt": "Describe this image.", "answer": '{"total": "9"}'},
        {"image": "2.png", "prompt": "What is it?", "answer": "a cat"},
    ]
    job = seen["job"]
    argv = job.argv
    assert argv[:3] == ("python", "-m", "vlm_pipeline.quantize.qat")
    assert argv[argv.index("--lr") + 1] == repr(1e-4)  # LoRA: x10 the full-FT preset
    assert argv[argv.index("--max-steps") + 1] == "20"
    assert argv[argv.index("--images") + 1] == "/workspace/data/mine"
    assert job.narrow
    assert job.read_only_project
    assert job.rw_dirs == (run,)
    assert seen["kw"]["est_peak_gb"] == est_peak_gb(w.catalog.get(MODEL).disk_gb, "lora")  # type: ignore[union-attr]
    assert out.outputs["quantized"].type == "QuantizedModel"


@pytest.mark.usefixtures("ready")
def test_hub_training_data_and_separate_calibration_are_refused() -> None:
    spec = REGISTRY.get("quantize.qat")
    assert spec is not None
    params = spec.validate_params({})
    for inputs, reason in (
        ({"train_data": "dataset:hub"}, "hub_train_data"),
        ({"train_data": "dataset:local", "calib_data": "dataset:local"}, "separate_calib_data"),
    ):
        with pytest.raises(qat.QatInputUnsupportedError) as err:
            qat.check_support("q", params, None, inputs)  # type: ignore[arg-type]
        assert err.value.code == "CFG-013"
        assert err.value.params["reason"] == reason
    qat.check_support("q", params, None, {"train_data": "dataset:local"})  # type: ignore[arg-type]


def test_full_adapter_needs_more_memory_than_lora() -> None:
    assert est_peak_gb(4.0, "full") > est_peak_gb(4.0, "lora") > 4.0 * 2


@pytest.mark.usefixtures("ready")
@pytest.mark.parametrize("case", ["link_out", "dotdot"])
def test_images_outside_the_folder_are_refused_before_any_file(tmp_path: Path, case: str) -> None:
    from vlm_pipeline.app.workers import LocalDatasetInvalidError

    w = RealWorker(_root(tmp_path))
    model, data = _model(w), _dataset(w, ["a"], [None])
    folder = w.root / "data" / "mine"
    records = w.root / "outputs" / "graph" / "runs" / "d" / "records.jsonl"
    if case == "link_out":
        (tmp_path / "secret.png").write_bytes(b"x")
        (folder / "0.png").unlink()
        (folder / "0.png").symlink_to(tmp_path / "secret.png")
    else:  # a tampered records.jsonl: the Record rule refuses ../
        row = json.loads(records.read_text())
        records.write_text(json.dumps({**row, "image": "../x.png"}) + "\n")
    job = _job(w, {}, {"model": model, "train_data": data})
    with pytest.raises(LocalDatasetInvalidError) as err:
        run_qat(w, job)
    assert err.value.params["first_reason"] == "bad_image_name"
    assert not (job.run_dir / DATA_NAME).exists()
