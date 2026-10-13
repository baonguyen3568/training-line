"""The default template ("Đọc hoá đơn", invoice_to_json) through the REAL executor and the REAL worker (tester, E2E).

Only what needs a GPU is faked: the train container (``workers.run_job``) and the eval engine
(``node_worker.run_eval_benchmarks``). Everything between them is the code the app runs: model pick, hub dataset,
split, recipe building, output path rules (``check_output_paths``), artifact hand-over between nodes, run.json.
The 2026-10-13 smoke failed at the first node on exactly this hand-over, which the per-node tests did not cover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import workers
from vlm_pipeline.app.workers import PREPARED_DIRS, RealWorker
from vlm_pipeline.artifacts.index import ArtifactIndex
from vlm_pipeline.common.catalog import load_dataset_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest
from vlm_pipeline.eval import node_worker
from vlm_pipeline.graph import validate
from vlm_pipeline.graph.executor import Executor
from vlm_pipeline.nodes import REGISTRY

TEMPLATE = project_root() / "configs" / "templates" / "invoice_to_json.json"


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")
    model = root / ".cache" / "prepared" / PREPARED_DIRS["nemotron-parse-v1.1"] / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    cord = next(d for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")
    snap = root / ".cache" / "hf" / "hub" / "datasets--naver-clova-ix--cord-v2" / "snapshots" / cord.revision
    snap.mkdir(parents=True)
    return root


def test_the_default_template_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    seen: dict[str, Any] = {}

    def fake_train(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen["train_argv"] = job.argv
        seen["config"] = (run_dir / "config.yaml").read_text()
        (run_dir / "checkpoints" / "LATEST" / "model" / "consolidated").mkdir(parents=True)
        final = manifest.model_copy(update={"status": RunStatus.COMPLETED, "exit_code": 0})
        write_manifest(run_dir, final)  # like run_job: the lineage (base model) later nodes read
        return final

    def fake_eval(request: node_worker.EvalRequest) -> node_worker.EvalOutcome:
        seen["eval_model"], seen["eval_base"] = request.model.path, request.base.path
        report = request.run_dir / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"gate": "PASS"}))
        return node_worker.EvalOutcome(report_path=report, report_hash="h", status="PASS")

    monkeypatch.setattr(workers, "run_job", fake_train)
    monkeypatch.setattr(node_worker, "run_eval_benchmarks", fake_eval)
    monkeypatch.setattr("vlm_pipeline.train.recipe_checks.resolve_parse_class_token_start_idx", lambda _p: 50000)
    monkeypatch.setattr("vlm_pipeline.train.ckpt_complete.complete_from_base", lambda base, out: [])

    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    workflow = template.get("workflow", template)
    worker = RealWorker(root)
    assert [i for i in validate(workflow, REGISTRY, workspace=root) if i.severity == "error"] == []
    run = Executor(REGISTRY, worker, root / "outputs" / "graph", index=ArtifactIndex(tmp_path / "i.db"),
                   workspace=root).run(workflow)  # fmt: skip

    assert {nid: (err.code, str(err)) for nid, err in run.errors.items()} == {}
    prepared = (root / ".cache" / "prepared" / PREPARED_DIRS["nemotron-parse-v1.1"] / "model").resolve()
    assert "pretrained_model_name_or_path: /workspace/.cache/prepared/" in seen["config"]  # base reached the recipe
    assert seen["eval_base"] == prepared
    assert str(seen["eval_model"]).endswith("checkpoints/LATEST/model/consolidated")
    for outputs in run.outputs.values():  # every hand-over is inside a node run dir (the executor rule)
        for ref in outputs.values():
            assert Path(ref.path).resolve().is_relative_to((root / "outputs" / "graph").resolve())
