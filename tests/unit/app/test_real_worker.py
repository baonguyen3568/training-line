from __future__ import annotations

import dataclasses
import json
import os
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
    assert ei.value.code == "NET-002"
    snap = _snap(root, "models", entry.repo, entry.revision)
    out = w.run(_job("model.hf_pick", {"model": "cosmos-reason2-2b"}, root / "r")).outputs["model"]
    assert out.type == "Model"
    assert out.path == str((root / "r" / "model.json").resolve())  # inside the run dir (executor rule)
    assert w.model_dir(out) == snap.resolve()  # consumers get the cached folder
    assert out.hash == f"{entry.repo}@{entry.revision}"


def test_model_pick_passes_the_executor_output_rule(tmp_path: Path) -> None:
    """E2E smoke 2026-10-13 failed at n1: the pick returned the cache folder, outside its run dir (tester)."""
    from vlm_pipeline.artifacts.index import ArtifactIndex
    from vlm_pipeline.graph.executor import Executor

    root = _root(tmp_path)
    w = RealWorker(root)
    entry = w.catalog.get("cosmos-reason2-2b")
    assert entry is not None
    snap = _snap(root, "models", entry.repo, entry.revision)
    node = {"id": "m", "type": "model.hf_pick", "version": "1.0.0", "params": {"model": entry.id}}
    wf = {"format": "vlm-workflow", "schema_version": 1, "name": "pick", "edges": [], "nodes": [node]}
    executor = Executor(REGISTRY, w, root / "outputs" / "graph", index=ArtifactIndex(tmp_path / "i.db"), workspace=root)
    run = executor.run(wf)
    assert run.errors == {}
    assert w.model_dir(run.outputs["m"]["model"]) == snap.resolve()


@pytest.mark.parametrize("bad", ["/etc", "../outside", "outputs/runs/x", ".cache/../src"])
def test_a_model_descriptor_cannot_point_outside_the_cache(tmp_path: Path, bad: str) -> None:
    root = _root(tmp_path)
    (root / "outputs" / "runs" / "x").mkdir(parents=True)
    (root / "src").mkdir()
    (tmp_path / "outside").mkdir()
    desc = root / "outputs" / "graph" / "runs" / "m-0123456789abcdef" / "model.json"
    desc.parent.mkdir(parents=True)
    desc.write_text(json.dumps({"format": "vlm-model-ref", "schema_version": 1, "repo": "a/b", "revision": "r",
                                "path": bad}))  # fmt: skip
    with pytest.raises(ValueError, match=r"under \.cache"):
        RealWorker(root).model_dir(ArtifactRef("Model", str(desc), "a/b@r"))


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
    got = w.dataset_input(ref)
    assert (got.path, got.split, got.limit) == (snapshot, "train", 100)


def test_split_uses_the_official_validation_split_and_keeps_cache_keys(tmp_path: Path) -> None:
    from vlm_pipeline.common.catalog import load_dataset_catalog

    root = _root(tmp_path)
    rev = next(d.revision for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")
    _snap(root, "datasets", "naver-clova-ix/cord-v2", rev)
    w = RealWorker(root)
    src = w.run(_job("data.hf_load", {"repo": "naver-clova-ix/cord-v2", "max_samples": 100}, root / "a"))
    out = w.run(_job("data.split", {}, root / "b", {"dataset": src.outputs["dataset"]})).outputs
    train, val = w.dataset_input(out["train"]), w.dataset_input(out["val"])
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
    from vlm_pipeline.common.launcher import LocalDockerLauncher

    monkeypatch.delenv("VLM_RUNNER_SOCKET", raising=False)
    monkeypatch.setattr(LocalDockerLauncher, "image_id", lambda self, image: "sha256:abc")
    w = RealWorker(_root(tmp_path), image="vlm-automodel:26.08-3")
    train, pick = REGISTRY.get("train.sft_vlm"), REGISTRY.get("model.hf_pick")
    assert train is not None
    assert pick is not None
    assert w.identity(train).startswith("container/vlm-automodel:26.08-3@sha256:abc+recipe:")
    assert w.identity(pick) == "host/real-1"


def test_identity_changes_with_any_template_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A template edit (e.g. processor pixel limits) must never reuse a cached run made with the old template."""
    import vlm_pipeline.common.jobrun as jobrun

    monkeypatch.setattr(jobrun, "image_digest", lambda image: "sha256:abc")
    import shutil

    from vlm_pipeline.common.paths import project_root

    root = tmp_path / "proj"  # a COPY of the templates: _root() symlinks the real ones
    shutil.copytree(project_root() / "configs" / "train" / "templates", root / "configs" / "train" / "templates")
    train = REGISTRY.get("train.lora_vlm")
    assert train is not None
    before = RealWorker(root, image="i").identity(train)
    assert RealWorker(root, image="i").identity(train) == before  # unchanged templates: same key
    template = next((root / "configs" / "train" / "templates").glob("*.yaml"))
    template.write_bytes(template.read_bytes() + b" ")
    assert RealWorker(root, image="i").identity(train) != before


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


def test_eval_resolves_the_base_of_a_trained_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.common.runinfo import write_manifest
    from vlm_pipeline.eval import node_worker
    from vlm_pipeline.eval.node_worker import EvalOutcome

    root = _root(tmp_path)
    w = RealWorker(root)
    parse = w.catalog.get("nemotron-parse-v1.1")
    assert parse is not None
    prepared = root / ".cache" / "prepared" / workers.PREPARED_DIRS[parse.id] / "model"
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    train_dir = root / "outputs" / "graph" / "runs" / "t-abc"
    ckpt = train_dir / "checkpoints" / "LATEST" / "model" / "consolidated"
    ckpt.mkdir(parents=True)
    write_manifest(train_dir, RunManifest(run_id="t-abc", name="t", stage="train", task="sft",
                                          base_model=parse.repo, base_model_revision=parse.revision))  # fmt: skip
    seen: dict[str, Any] = {}

    def fake_eval(request: Any) -> EvalOutcome:
        seen["request"] = request
        report = request.run_dir / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}")
        return EvalOutcome(report, "h", "PASS")

    monkeypatch.setattr(node_worker, "run_eval_benchmarks", fake_eval)
    model = ArtifactRef("Checkpoint", str(ckpt), "t-abc:cfg")
    out = w.run(_job("eval.benchmarks", {}, root / "outputs" / "graph" / "runs" / "e-1", {"model": model}))
    req = seen["request"]
    assert (req.entry.id, req.base.hash, req.model.hash) == (parse.id, f"{parse.repo}@{parse.revision}", "t-abc:cfg")
    assert req.base.path == prepared.resolve()
    assert out.outputs["report"].type == "EvalReport"
    assert w.supports(REGISTRY.get("eval.benchmarks"))  # type: ignore[arg-type]


def test_eval_of_a_catalog_model_compares_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.eval import node_worker
    from vlm_pipeline.eval.node_worker import EvalOutcome

    root = _root(tmp_path)
    w = RealWorker(root)
    entry = w.catalog.get("nemotron-parse-v1.1")  # the model the eval engine supports today
    assert entry is not None
    prepared = root / ".cache" / "prepared" / workers.PREPARED_DIRS[entry.id] / "model"
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    seen: dict[str, Any] = {}

    def fake_eval(req: Any) -> EvalOutcome:
        seen["req"] = req
        return EvalOutcome(req.run_dir / "r.json", "h", "PASS")

    monkeypatch.setattr(node_worker, "run_eval_benchmarks", fake_eval)
    model = ArtifactRef("Model", str(prepared), f"{entry.repo}@{entry.revision}")
    w.run(_job("eval.benchmarks", {}, root / "outputs" / "graph" / "runs" / "e-2", {"model": model}))
    assert seen["req"].model.hash == seen["req"].base.hash  # the engine then skips the base comparison


def test_eval_of_an_untraceable_checkpoint_runs_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.eval import node_worker

    root = _root(tmp_path)
    w = RealWorker(root)
    (tmp_path / "run.json").write_text("{}")  # a stray record ABOVE the project must never be read
    ckpt = root / "outputs" / "graph" / "runs" / "lost" / "model"
    ckpt.mkdir(parents=True)
    monkeypatch.setattr(node_worker, "run_eval_benchmarks", lambda req: pytest.fail("no job may start"))
    with pytest.raises(workers.UnknownModelError):
        w.run(_job("eval.benchmarks", {}, root / "outputs" / "graph" / "runs" / "e-3",
                   {"model": ArtifactRef("Checkpoint", str(ckpt), "lost:cfg")}))  # fmt: skip


def test_local_folder_split_and_train_use_meta_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pil = pytest.importorskip("PIL.Image")
    root = _root(tmp_path)
    data = root / "data" / "mine"
    data.mkdir(parents=True)
    lines = []
    for i in range(40):
        pil.new("RGB", (32, 24)).save(data / f"r{i}.png")
        lines.append(json.dumps({"image": f"r{i}.png", "target": {"total": str(i)}}))
    (data / "labels.jsonl").write_text("\n".join(lines) + "\n")
    w = RealWorker(root)
    src = w.run(_job("data.local_folder", {"path": "data/mine"}, root / "outputs" / "graph" / "runs" / "lf"))
    parts = w.run(_job("data.split", {"val_fraction": 0.2, "seed": 3}, root / "outputs" / "graph" / "runs" / "sp",
                       {"dataset": src.outputs["dataset"]})).outputs  # fmt: skip
    assert parts["train"].hash.startswith("local:")
    assert "#split=train#val=0.2#seed=3" in parts["train"].hash
    monkeypatch.setattr("vlm_pipeline.train.recipe_checks.resolve_parse_class_token_start_idx", lambda _p: 50000)
    monkeypatch.setattr(
        workers, "run_job", lambda job, m, d, **kw: m.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})
    )
    parse = w.catalog.get("nemotron-parse-v1.1")
    assert parse is not None
    base = ArtifactRef("Model", str(root / "m"), f"{parse.repo}@{parse.revision}")
    run_dir = root / "outputs" / "graph" / "runs" / "t"
    w.run(
        _job("train.sft_vlm", {}, run_dir, {"base_model": base, "train_data": parts["train"], "val_data": parts["val"]})
    )
    cfg = (run_dir / "config.yaml").read_text()
    assert cfg.count("make_meta_dataset") == 2
    assert "/workspace/outputs/graph/runs/t/data/train.meta.json" in cfg
    train_lines = (run_dir / "data" / "train.jsonl").read_text().splitlines()
    val_lines = (run_dir / "data" / "validation.jsonl").read_text().splitlines()
    assert len(train_lines) + len(val_lines) == 40
    first = json.loads(train_lines[0])
    assert first["messages"][1]["content"].startswith("<s_total>")  # Parse: json2token target
    meta = json.loads((run_dir / "data" / "train.meta.json").read_text())
    assert meta["train"]["media_dir"] == "/workspace/data/mine"


def test_the_lock_owner_says_who_runs(tmp_path: Path) -> None:
    """PM: a script driving RealWorker must not look like the app in GPU_LOCK."""
    import os

    assert RealWorker(_root(tmp_path), who="vlm-app").who == "vlm-app"
    script = RealWorker(_root(tmp_path / "b")).who
    assert script.startswith("script:")
    assert script.endswith(f"/{os.getpid()}")


def test_a_retry_resumes_from_the_earlier_attempt_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P017: the new attempt's recipe restores from the executor-chosen checkpoint of the earlier attempt."""
    root = _root(tmp_path)
    w = RealWorker(root)
    seen: list[RunManifest] = []

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.append(manifest)
        return manifest.model_copy(update={"status": RunStatus.FAILED, "exit_code": 1})

    monkeypatch.setattr(workers, "run_job", fake_run_job)
    earlier = root / "outputs" / "graph" / "runs" / "t-0123456789abcdef" / "checkpoints" / "epoch_0_step_9"
    earlier.mkdir(parents=True)
    spec = REGISTRY.get("train.lora_vlm")
    assert spec is not None
    run_dir = root / "outputs" / "graph" / "runs" / "t-0123456789abcdef-a2"
    job = NodeJob("t", spec, spec.validate_params({"epochs": 1}), _train_inputs(w), run_dir, resume_from_step=9,
                  resume_from=earlier)  # fmt: skip
    with pytest.raises(ContainerExit):
        w.run(job)
    cfg = (run_dir / "config.yaml").read_text()
    assert "restore_from: /workspace/outputs/graph/runs/t-0123456789abcdef/checkpoints/epoch_0_step_9" in cfg
    assert "checkpoint_dir: /workspace/outputs/graph/runs/t-0123456789abcdef-a2/checkpoints" in cfg
    assert seen[0].resumed_from == {"run": "t-0123456789abcdef", "checkpoint": "epoch_0_step_9"}  # C6 lease key


def test_train_on_a_data_eval_folder_is_refused_before_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DATA-007 at run time (the canvas rule's twin): benchmark test data under data/eval never trains."""
    from vlm_pipeline.common.heldout_data import HeldoutDataInTrainingError

    pil = pytest.importorskip("PIL.Image")
    root = _root(tmp_path)
    data = root / "data" / "eval" / "ocrbench-92a54bd1"
    data.mkdir(parents=True)
    pil.new("RGB", (32, 24)).save(data / "a.png")
    (data / "labels.jsonl").write_text(json.dumps({"image": "a.png", "target": "x"}) + "\n")
    (root / "data" / "mine").symlink_to(data)  # a symlink into data/eval is the same folder
    w = RealWorker(root)
    src = w.run(_job("data.local_folder", {"path": "data/mine"}, root / "outputs" / "graph" / "runs" / "lf"))
    monkeypatch.setattr(workers, "run_job", lambda *a, **k: pytest.fail("no job may start"))
    qwen = w.catalog.get("qwen3.5-0.8b")
    assert qwen is not None
    base = ArtifactRef("Model", str(root / "m"), f"{qwen.repo}@{qwen.revision}")
    train = _job("train.lora_vlm", {}, root / "outputs" / "graph" / "runs" / "t", {"base_model": base,
                 "train_data": src.outputs["dataset"]})  # fmt: skip
    with pytest.raises(HeldoutDataInTrainingError) as err:
        w.run(train)
    assert err.value.code == "DATA-007"


@pytest.mark.parametrize("kind", ["file", "link", "fifo"])
def test_failure_log_tail_never_follows_a_job_made_link_or_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The run dir is the job's: a stdout.log swapped for a link to a secret (or a FIFO) gives an empty tail."""
    root = _root(tmp_path)
    secret = root / ".cache" / "hf" / "token"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("hf_SECRETSECRET\n")
    w = RealWorker(root)
    run_dir = root / "outputs" / "graph" / "g" / "n"

    def failing(job: Any, m: RunManifest, d: Path, **kw: Any) -> RunManifest:
        log = d / "logs" / "stdout.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        if kind == "file":
            log.write_text("step 1\nboom\n")
        elif kind == "link":
            log.symlink_to(secret)
        else:
            os.mkfifo(log)
        return m.model_copy(update={"status": RunStatus.FAILED, "exit_code": 1})

    monkeypatch.setattr(workers, "run_job", failing)
    with pytest.raises(ContainerExit) as ei:
        w.run(_job("train.lora_vlm", {}, run_dir, _train_inputs(w)))
    assert ei.value.log_tail == ("step 1\nboom" if kind == "file" else "")


def test_train_refuses_a_record_image_that_resolves_outside_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bao-security R1: an image swapped for a link out of the folder after the scan fails early, with its line."""
    pil = pytest.importorskip("PIL.Image")
    root = _root(tmp_path)
    data = root / "data" / "mine"
    data.mkdir(parents=True)
    pil.new("RGB", (32, 24)).save(data / "a.png")
    (data / "labels.jsonl").write_text(json.dumps({"image": "a.png", "target": "x"}) + "\n")
    w = RealWorker(root)
    src = w.run(_job("data.local_folder", {"path": "data/mine"}, root / "outputs" / "graph" / "runs" / "lf"))
    outside = tmp_path / "secret.png"
    pil.new("RGB", (32, 24)).save(outside)
    (data / "a.png").unlink()
    (data / "a.png").symlink_to(outside)
    monkeypatch.setattr(workers, "run_job", lambda *a, **k: pytest.fail("no job may start"))
    qwen = w.catalog.get("qwen3.5-0.8b")
    assert qwen is not None
    base = ArtifactRef("Model", str(root / "m"), f"{qwen.repo}@{qwen.revision}")
    job = _job("train.lora_vlm", {}, root / "outputs" / "graph" / "runs" / "t",
               {"base_model": base, "train_data": src.outputs["dataset"]})  # fmt: skip
    with pytest.raises(workers.LocalDatasetInvalidError) as err:
        w.run(job)
    assert (err.value.params["first_reason"], err.value.params["node"]) == ("bad_image_name", job.node_id)


def test_the_merge_estimate_fits_measured_peaks_and_lets_9b_run() -> None:
    """research 2026-10-13 (26.08-6): 2B fp32 merge peaked 9.5 GB, 4B bf16 8.5 GB; 9B was refused at 55 GB."""
    from vlm_pipeline.app.workers import _merge_peak_gb
    from vlm_pipeline.common.catalog import load_catalog

    models = {m.id: m for m in load_catalog().models}
    assert _merge_peak_gb(models["qwen3.5-2b"]) >= 9.5
    assert _merge_peak_gb(models["qwen3.5-4b"]) >= 8.5
    assert _merge_peak_gb(models["qwen3.5-9b"]) <= 33.1  # never more than the 9B training it follows
    assert _merge_peak_gb(models["qwen3.5-0.8b"]) == 10.0  # floor


def test_grpo_is_coming_soon_with_its_real_reason() -> None:
    """PM 2026-10-14: GRPO is refused (canvas and run share this check) with the engine's real limitation."""
    from types import SimpleNamespace

    from vlm_pipeline.app.workers import GRPO_BLOCKED_REASON, NodeNotRunnableError, _check_grpo, _support_checks

    assert _support_checks()["train.grpo_vlm"] is _check_grpo
    query = SimpleNamespace(node_id="g", spec=SimpleNamespace(type_id="train.grpo_vlm"), params=None, model=None,
                            inputs={})  # fmt: skip
    with pytest.raises(NodeNotRunnableError) as refused:
        _check_grpo(query)  # type: ignore[arg-type]
    assert refused.value.code == "CFG-011"
    assert refused.value.params == {"node": "g", "node_type": "train.grpo_vlm", "reason": GRPO_BLOCKED_REASON}
