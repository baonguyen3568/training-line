"""B-6 bench.perf host side (no GPU): what is measured, against what, and how the report comes back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import bench_worker
from vlm_pipeline.app.bench_worker import BenchUnsupportedError, run_bench
from vlm_pipeline.app.workers import RealWorker
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY

PARSE = "nemotron-parse-v1.1"


def _root(tmp_path: Path) -> Path:
    from vlm_pipeline.common.paths import project_root

    root = tmp_path / "proj"
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")
    prepared = root / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    return root


def _job(params: dict[str, Any], run_dir: Path, model: ArtifactRef) -> NodeJob:
    spec = REGISTRY.get("bench.perf")
    assert spec is not None
    return NodeJob(node_id="bench", spec=spec, params=spec.validate_params(params), inputs={"model": model},
                   run_dir=run_dir)  # fmt: skip


def _parse_base(w: RealWorker) -> ArtifactRef:
    entry = w.catalog.get(PARSE)
    assert entry is not None
    path = w.root / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
    return ArtifactRef("Model", str(path.resolve()), f"{entry.repo}@{entry.revision}")


def _trained(w: RealWorker) -> ArtifactRef:
    entry = w.catalog.get(PARSE)
    assert entry is not None
    run = w.root / "outputs" / "runs" / "train-1"
    ckpt = run / "checkpoints" / "LATEST" / "model" / "consolidated"
    ckpt.mkdir(parents=True)
    manifest = RunManifest(run_id="train-1", name="t", stage="train", task="sft", base_model=entry.repo,
                           base_model_revision=entry.revision)  # fmt: skip
    write_manifest(run, manifest)
    return ArtifactRef("Checkpoint", str(ckpt.resolve()), "train-1:abc")


class FakeJobs:
    def __init__(self, e2e_ms: dict[str, float], status: RunStatus = RunStatus.COMPLETED) -> None:
        self.e2e_ms = e2e_ms  # job name -> p50
        self.status = status
        self.calls: list[tuple[Any, dict[str, Any], Path]] = []

    def __call__(self, container: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        self.calls.append((container, kw, run_dir))
        name = run_dir.name
        if self.status == RunStatus.COMPLETED:
            result = {
                "e2e_ms_p50": self.e2e_ms[name], "tokens_per_s": 40.0, "memory_peak_load_gb": 6.0,
                "per_request": [], "decoding": {"max_new_tokens": 256, "repetition_penalty": 1.0},
                "overrides": {}, "model_class": "x", "conditions": {"engine": "hf", "batch_size": 1},
                "bench_sha256": "s", "host_baseline_available_gb": 50.0, "request": {"image_ids": [0, 1]},
            }  # fmt: skip
            (run_dir / "result.json").write_text(json.dumps(result))
        return manifest.model_copy(update={"status": self.status, "exit_code": 0, "container_digest": "sha256:img"})


def test_trained_model_vs_base_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    jobs = FakeJobs({"model": 900.0, "base": 1000.0})
    monkeypatch.setattr(bench_worker, "run_job", jobs)
    run_dir = w.root / "outputs" / "graph" / "runs" / "bench-1"
    out = run_bench(w, _job({}, run_dir, _trained(w))).outputs["report"]
    assert out.type == "BenchReport"
    assert out.hash.startswith("sha256:")
    report = json.loads(Path(out.path).read_text())
    assert report["ratio_vs_base"]["e2e_ms_p50"] == 0.9
    variants = {(r["variant"], r["metric"]) for r in report["records"]}
    assert ("model", "e2e_ms_p50") in variants
    assert ("base", "tokens_per_s") in variants
    assert all(r["benchmark"] == "perf" for r in report["records"])
    conditions = report["records"][0]["perf_conditions"]
    assert (conditions["image_ids"], conditions["bench_sha256"]) == ([0, 1], "s")  # comparable across runs
    assert [run_dir.name for _, _, run_dir in jobs.calls] == ["model", "base"]


def test_container_runs_the_trial_code_path_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    jobs = FakeJobs({"model": 900.0})
    monkeypatch.setattr(bench_worker, "run_job", jobs)
    run_bench(w, _job({"compare_with": "none", "num_requests": 20, "output_tokens": 128}, w.root / "r", _trained(w)))
    [(container, kw, run_dir)] = jobs.calls
    assert container.argv[:3] == ("python", "-m", "vlm_pipeline.serve.bench")
    assert container.read_only_project
    assert container.rw_dirs == (run_dir,)
    assert kw["require_approved_image"] is True
    assert "on_wait" in kw
    assert "cancel" in kw
    request = json.loads((run_dir / "request.json").read_text())
    assert request["num_requests"] == 20
    assert request["trial"]["max_new_tokens"] == 128
    assert request["trial"]["decode_overrides"] == {"repetition_penalty": 1.0}  # same decoding as Chạy thử / eval
    assert request["trial"]["generation_config_dir"].endswith("nemotron-parse-v1.1_fd1e3c6b/model")


def test_benchmarking_the_base_itself_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    jobs = FakeJobs({"model": 1000.0})
    monkeypatch.setattr(bench_worker, "run_job", jobs)
    out = run_bench(w, _job({}, w.root / "r", _parse_base(w))).outputs["report"]
    assert len(jobs.calls) == 1
    assert "ratio_vs_base" not in json.loads(Path(out.path).read_text())


@pytest.mark.parametrize("params", [{"scenario": "multi_user"}, {"compare_with": "unquantized"}])
def test_unsupported_requests_fail_before_any_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    w = RealWorker(_root(tmp_path))
    jobs = FakeJobs({})
    monkeypatch.setattr(bench_worker, "run_job", jobs)
    with pytest.raises(BenchUnsupportedError) as exc:
        run_bench(w, _job(params, w.root / "r", _trained(w)))
    assert exc.value.code == "CFG-011"
    assert jobs.calls == []


def test_failed_job_is_a_container_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    monkeypatch.setattr(bench_worker, "run_job", FakeJobs({}, RunStatus.FAILED))
    with pytest.raises(ContainerExit):
        run_bench(w, _job({}, w.root / "r", _trained(w)))


@pytest.mark.parametrize(
    ("model_id", "params", "ok"),
    [(PARSE, {}, True), (PARSE, {"scenario": "multi_user"}, False), (PARSE, {"compare_with": "unquantized"}, False),
     ("qwen3.5-0.8b", {}, False)],
)  # fmt: skip
def test_check_support_matches_the_worker(model_id: str, params: dict[str, Any], ok: bool) -> None:
    from vlm_pipeline.app.bench_worker import check_support
    from vlm_pipeline.common.catalog import load_catalog
    from vlm_pipeline.serve.config import BenchParams

    model = load_catalog().get(model_id)
    if ok:
        check_support("bench", BenchParams(**params), model, {"model": "model"})
    else:
        with pytest.raises(BenchUnsupportedError) as exc:
            check_support("bench", BenchParams(**params), model, {"model": "model"})
        assert exc.value.params == {"node": "bench", "node_type": "bench.perf"}
    check_support("bench", BenchParams(scenario="multi_user"), None, {})  # unwired model: validate's CFG-003
