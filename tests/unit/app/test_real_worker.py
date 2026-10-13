from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import workers
from vlm_pipeline.app.workers import NotCachedError, RealWorker
from vlm_pipeline.common import images
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
    snapshot = f"/workspace/.cache/hf/hub/datasets--naver-clova-ix--cord-v2/snapshots/{rev}"
    got = w._dataset_input(ref)
    assert (got.path, got.split, got.limit) == (snapshot, "train", 100)


def test_split_uses_the_official_validation_split_and_keeps_cache_keys(tmp_path: Path) -> None:
    from vlm_pipeline.common.catalog import load_dataset_catalog

    root = _root(tmp_path)
    rev = next(d.revision for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")
    _snap(root, "datasets", "naver-clova-ix/cord-v2", rev)
    w = RealWorker(root)
    src = w.run(_job("data.hf_load", {"repo": "naver-clova-ix/cord-v2", "max_samples": 100}, root / "a"))
    out = w.run(_job("data.split", {}, root / "b", {"dataset": src.outputs["dataset"]})).outputs
    train, val = w._dataset_input(out["train"]), w._dataset_input(out["val"])
    assert (train.split, train.limit, val.split, val.limit) == ("train", 100, "validation", 0)
    assert out["train"].hash == src.outputs["dataset"].hash  # train split = the loaded data: same cache key
    assert out["val"].hash == f"naver-clova-ix/cord-v2@{rev}#split=validation"


def test_train_reads_the_split_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common.dataset_ref import DatasetRef, write_dataset_ref

    root = _root(tmp_path)
    w = RealWorker(root)
    monkeypatch.setattr(
        workers, "run_job", lambda job, m, d, **kw: m.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})
    )
    refs = {}
    for name, split, limit in (("train_data", "train", 100), ("val_data", "validation", 0)):
        path = root / "d" / f"{name}.json"
        write_dataset_ref(
            path, DatasetRef(repo="r/x", revision="0" * 40, path="/workspace/ds", split=split, limit=limit)
        )
        refs[name] = ArtifactRef("Dataset", str(path), name)
    monkeypatch.setattr("vlm_pipeline.train.recipe_checks.resolve_parse_class_token_start_idx", lambda _path: 50000)
    parse = w.catalog.get("nemotron-parse-v1.1")
    assert parse is not None
    base = ArtifactRef("Model", str(root / "m"), f"{parse.repo}@{parse.revision}")
    run_dir = root / "outputs" / "graph" / "g" / "n"
    w.run(_job("train.sft_vlm", {}, run_dir, {"base_model": base, **refs}))
    cfg = (run_dir / "config.yaml").read_text()
    assert "limit_dataset_samples: 100" in cfg
    assert "split: validation" in cfg


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


_ROW = "vlm-automodel | {tag} (local) | (local) | base arm64 sha256:aaa | {id} | 29GB | {review} | x"
LOCK = "\n".join(
    [
        "# header",
        _ROW.format(tag="vlm-automodel:26.08-5", id="sha256:111 (rebuilt)", review="bao-security APPROVED (x)"),
        _ROW.format(tag="vlm-automodel:26.08-4", id="sha256:222", review="bao-security APPROVED"),
    ]
)


def test_pinned_image_id_parsing() -> None:
    assert images.pinned_image_id(LOCK, "vlm-automodel:26.08-5") == "sha256:111"
    assert images.pinned_image_id(LOCK, "vlm-automodel:26.08-4") == "sha256:222"
    assert images.pinned_image_id(LOCK, "vlm-automodel:9") is None


def test_check_pinned_image(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "IMAGES.lock").write_text(LOCK)
    images.check_pinned_image(tmp_path, "vlm-automodel:26.08-5", "sha256:111")
    for actual in ("sha256:999", None):
        with pytest.raises(images.ImageMismatchError) as ei:
            images.check_pinned_image(tmp_path, "vlm-automodel:26.08-5", actual)
        assert ei.value.code == "SEC-004"
    with pytest.raises(images.ImageMismatchError):
        images.check_pinned_image(tmp_path, "vlm-automodel:unpinned", "sha256:111")


def test_real_lock_pins_default_image() -> None:
    from vlm_pipeline.common.paths import project_root

    text = (project_root() / "docker" / "IMAGES.lock").read_text()
    assert images.pinned_image_id(text, workers.DEFAULT_TRAIN_IMAGE) is not None


def test_user_jobs_require_approved_image(tmp_path: Path) -> None:
    lock = "\n".join(
        [
            LOCK,
            _ROW.format(tag="vlm-automodel:probe", id="sha256:333", review="PROBE, not approved"),
            _ROW.format(tag="vlm-automodel:u", id="sha256:444", review="bao-security UNAPPROVED"),
            _ROW.format(tag="vlm-automodel:d", id="sha256:555", review="DISAPPROVED by bao-security APPROVED"),
            _ROW.format(tag="vlm-automodel:p", id="sha256:666", review="bao-security APPROVED-PENDING"),
        ]
    )
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "IMAGES.lock").write_text(lock)
    images.check_pinned_image(tmp_path, "vlm-automodel:probe", "sha256:333")  # dev/probe path ok
    with pytest.raises(images.ImageMismatchError):
        images.check_pinned_image(tmp_path, "vlm-automodel:probe", "sha256:333", require_approved=True)
    assert images.is_approved(lock, "vlm-automodel:26.08-5")
    assert not images.is_approved(lock, "vlm-automodel:probe")
    assert not images.is_approved(lock, "vlm-automodel:u")
    assert not images.is_approved(lock, "vlm-automodel:d")
    assert not images.is_approved(lock, "vlm-automodel:p")


def test_full_finetune_output_is_completed_from_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vlm_pipeline.train.ckpt_complete as ckpt_complete

    root = _root(tmp_path)
    w = RealWorker(root)
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(ckpt_complete, "complete_from_base", lambda base, out: calls.append((base, out)) or [])

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        (run_dir / "checkpoints" / "LATEST" / "model" / "consolidated").mkdir(parents=True)
        return manifest.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})

    monkeypatch.setattr(workers, "run_job", fake_run_job)
    run_dir = root / "outputs" / "graph" / "runs" / "t-x"
    monkeypatch.setattr("vlm_pipeline.train.recipe_checks.resolve_parse_class_token_start_idx", lambda _path: 50000)
    parse = w.catalog.get("nemotron-parse-v1.1")
    assert parse is not None
    inputs = {**_train_inputs(w), "base_model": ArtifactRef("Model", str(root / "m"), f"{parse.repo}@{parse.revision}")}
    out = w.run(_job("train.sft_vlm", {"epochs": 1}, run_dir, inputs)).outputs["model"]
    assert out.path.endswith("checkpoints/LATEST/model/consolidated")
    assert calls == [(w.root / "m", Path(out.path))]


def test_lora_on_full_finetune_only_model_is_refused_before_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    w = RealWorker(root)
    entry = w.catalog.get("nemotron-parse-v1.1")
    assert entry is not None
    assert entry.train_node == "train.sft_vlm"
    monkeypatch.setattr(workers, "run_job", lambda *a, **kw: pytest.fail("job must not start"))
    inputs = {**_train_inputs(w), "base_model": ArtifactRef("Model", str(root / "m"), f"{entry.repo}@{entry.revision}")}
    with pytest.raises(workers.TrainModeUnsupportedError) as ei:
        w.run(_job("train.lora_vlm", {}, root / "outputs" / "graph" / "g" / "n", inputs))
    assert ei.value.code == "CFG-010"
    assert ei.value.params == {"node": "n", "model": entry.display_name, "supported": "train.sft_vlm"}


def test_full_finetune_on_lora_only_model_is_refused_before_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = RealWorker(_root(tmp_path))
    monkeypatch.setattr(workers, "run_job", lambda *a, **kw: pytest.fail("job must not start"))
    with pytest.raises(workers.TrainModeUnsupportedError) as ei:  # cosmos-reason2-2b: catalog train_node lora_vlm
        w.run(_job("train.sft_vlm", {}, w.root / "outputs" / "graph" / "g" / "n", _train_inputs(w)))
    assert ei.value.params["supported"] == "train.lora_vlm"


def test_resume_of_unresumable_model_starts_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    w = RealWorker(root)
    entry = w.catalog.get("nemotron-parse-v1.1")
    assert entry is not None
    monkeypatch.setattr(workers, "RESUME_UNSUPPORTED", frozenset({entry.id}))  # mechanism kept, list empty now
    monkeypatch.setattr("vlm_pipeline.train.recipe_checks.resolve_parse_class_token_start_idx", lambda _path: 50000)
    run_dir = root / "outputs" / "graph" / "g" / "n"
    (run_dir / "checkpoints" / "epoch_0_step_1").mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("{}\n")
    seen: dict[str, bool] = {}

    def fake_run_job(job: Any, manifest: RunManifest, d: Path, **kw: Any) -> RunManifest:
        seen["fresh"] = not (d / "checkpoints").exists() and not (d / "metrics.jsonl").exists()
        return manifest.model_copy(update={"status": RunStatus.FAILED, "exit_code": 1})

    monkeypatch.setattr(workers, "run_job", fake_run_job)
    inputs = {**_train_inputs(w), "base_model": ArtifactRef("Model", str(root / "m"), f"{entry.repo}@{entry.revision}")}
    job = dataclasses.replace(_job("train.sft_vlm", {}, run_dir, inputs), resume_from_step=1)
    with pytest.raises(ContainerExit):
        w.run(job)
    assert seen == {"fresh": True}
    kept = sorted(p.name.split(".not-resumable-")[0] for p in run_dir.glob("*.not-resumable-*"))
    assert kept == ["checkpoints", "metrics.jsonl"]
    assert (run_dir / workers.NOT_RESUMABLE_MARKER).exists()  # executor: never offer "resume" for this run dir


def test_parse_resumes_since_patch_0005() -> None:
    """BUG-003 closed (26.08-6): no model starts over anymore; the error card offers "resume" again for Parse."""
    assert frozenset() == workers.RESUME_UNSUPPORTED
    assert workers.DEFAULT_TRAIN_IMAGE == "vlm-automodel:26.08-6"


def test_merge_lora_runs_on_cpu_and_completes_from_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vlm_pipeline.train.ckpt_complete as ckpt_complete

    root = _root(tmp_path)
    w = RealWorker(root)
    seen: dict[str, Any] = {}
    completed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(ckpt_complete, "complete_from_base", lambda base, out: completed.append((base, out)) or [])

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.update(job=job, manifest=manifest)
        return manifest.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})

    monkeypatch.setattr(workers, "run_job", fake_run_job)
    entry = w.catalog.get("qwen3.5-0.8b")
    assert entry is not None
    inputs = {
        "base_model": ArtifactRef("Model", str(root / "m"), f"{entry.repo}@{entry.revision}"),
        "adapter": ArtifactRef("Adapter", str(root / "outputs" / "graph" / "runs" / "t-1" / "a"), "t-1:abc"),
    }
    run_dir = root / "outputs" / "graph" / "runs" / "merge-1"
    out = w.run(_job("train.merge_lora", {}, run_dir, inputs)).outputs["model"]
    job = seen["job"]
    assert (job.gpu, job.read_only_project, job.rw_dirs) == (False, True, (run_dir.resolve(),))
    assert job.argv[:3] == ("python", "-m", "vlm_pipeline.train.merge_lora")
    assert "--auto-class" in job.argv
    assert seen["manifest"].parent_run_id == "t-1"
    assert (out.type, out.path) == ("Checkpoint", str(run_dir.resolve() / "merged"))
    assert completed == [(root / "m", run_dir.resolve() / "merged")]
