from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import workers
from vlm_pipeline.app.workers import NotCachedError, RealWorker
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY

COSMOS = "nvidia/Cosmos-Reason2-2B"


def _root(tmp_path: Path) -> Path:
    from vlm_pipeline.common.paths import project_root

    root = tmp_path / "proj"
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")
    return root


def _job(type_id: str, params: dict[str, Any], run_dir: Path, inputs: dict[str, ArtifactRef] | None = None) -> NodeJob:
    spec = REGISTRY.get(type_id)
    assert spec is not None
    return NodeJob(node_id="n", spec=spec, params=spec.validate_params(params), inputs=inputs or {}, run_dir=run_dir)


def _snap(root: Path, kind: str, repo: str, rev: str) -> Path:
    d = root / ".cache" / "hf" / "hub" / f"{kind}--{repo.replace('/', '--')}" / "snapshots" / rev
    d.mkdir(parents=True)
    return d


def test_model_pick_requires_cached_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    w = RealWorker(root)
    entry = w.catalog.get("cosmos-reason2-2b")
    assert entry is not None
    with pytest.raises(NotCachedError) as ei:
        w.run(_job("model.hf_pick", {"model": "cosmos-reason2-2b"}, root / "r"))
    assert ei.value.code == "NET-001"
    snap = _snap(root, "models", entry.repo, entry.revision)
    out = w.run(_job("model.hf_pick", {"model": "cosmos-reason2-2b"}, root / "r")).outputs["model"]
    assert out.type == "Model"
    assert out.path == str(snap.resolve())  # host path: executor/retention verify it on the host
    assert out.hash == f"{entry.repo}@{entry.revision}"


def test_hub_dataset_pinned(tmp_path: Path) -> None:
    root = _root(tmp_path)
    from vlm_pipeline.common.catalog import load_dataset_catalog

    rev = next(d.revision for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")
    _snap(root, "datasets", "naver-clova-ix/cord-v2", rev)
    w = RealWorker(root)
    out = w.run(_job("data.hf_load", {"repo": "naver-clova-ix/cord-v2"}, root / "r")).outputs
    assert out["dataset"].hash == f"naver-clova-ix/cord-v2@{rev}"
    limited = w.run(_job("data.hf_load", {"repo": "naver-clova-ix/cord-v2", "max_samples": 100}, root / "r2"))
    ref = limited.outputs["dataset"]
    assert ref.hash.endswith("#limit=100")
    assert w._dataset_input(ref) == (f"/workspace/.cache/hf/hub/datasets--naver-clova-ix--cord-v2/snapshots/{rev}", 100)


def _train_inputs(w: RealWorker) -> dict[str, ArtifactRef]:
    entry = w.catalog.get("cosmos-reason2-2b")
    assert entry is not None
    return {
        "base_model": ArtifactRef("Model", str(w.root / "m"), f"{entry.repo}@{entry.revision}"),
        "train_data": ArtifactRef("Dataset", str(w.root / "d"), "naver-clova-ix/cord-v2@x"),
    }


def test_train_builds_recipe_and_runs_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    w = RealWorker(root)
    seen: dict[str, Any] = {}

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.update(job=job, kw=kw)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "training.jsonl").write_text(
            json.dumps({"step": 0, "epoch": 0, "timestamp": "2026-10-12T00:00:00Z", "loss": 1.0, "mem": 3.0}) + "\n"
        )
        return manifest.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})

    monkeypatch.setattr(workers, "run_job", fake_run_job)
    run_dir = root / "outputs" / "graph" / "g1" / "n"
    out = w.run(_job("train.lora_vlm", {"epochs": 1}, run_dir, _train_inputs(w))).outputs["adapter"]
    cfg = (run_dir / "config.yaml").read_text()
    assert "pretrained_model_name_or_path: /workspace/m" in cfg
    assert "path_or_dataset: /workspace/d" in cfg
    assert seen["job"].argv[0] == "automodel"
    assert seen["kw"]["est_peak_gb"] > 0
    assert isinstance(seen["kw"]["cancel"], threading.Event)
    assert out.type == "Adapter"
    assert out.path == str(run_dir.resolve() / "checkpoints" / "LATEST" / "model")
    assert json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])["mem_max_alloc_gb"] == 3.0


@pytest.mark.parametrize(
    ("status", "exit_code", "expected_exit", "cancelled"),
    [(RunStatus.INTERRUPTED, 137, 75, False), (RunStatus.CANCELLED, 143, 143, True), (RunStatus.FAILED, 1, 1, False)],
)
def test_train_failure_raises_container_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: RunStatus, exit_code: int, expected_exit: int,
    cancelled: bool,
) -> None:  # fmt: skip
    root = _root(tmp_path)
    w = RealWorker(root)
    monkeypatch.setattr(
        workers, "run_job", lambda job, m, d, **kw: m.model_copy(update={"status": status, "exit_code": exit_code})
    )
    with pytest.raises(ContainerExit) as ei:
        w.run(_job("train.lora_vlm", {}, root / "outputs" / "graph" / "g" / "n", _train_inputs(w)))
    assert ei.value.exit_code == expected_exit
    assert ei.value.cancelled_by_executor is cancelled


def test_identity_keys_container_nodes_by_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vlm_pipeline.common.jobrun as jobrun

    monkeypatch.setattr(jobrun, "image_digest", lambda image: "sha256:abc")
    w = RealWorker(_root(tmp_path), image="vlm-automodel:26.08-3")
    train, pick = REGISTRY.get("train.sft_vlm"), REGISTRY.get("model.hf_pick")
    assert train is not None
    assert pick is not None
    assert w.identity(train) == "container/vlm-automodel:26.08-3@sha256:abc"
    assert w.identity(pick) == "host/real-1"
