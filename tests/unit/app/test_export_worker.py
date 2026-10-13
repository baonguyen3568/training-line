"""B-2 export (native models, ADR-012): portable folder, licences + notices, fail-closed on remote code and paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import export_worker
from vlm_pipeline.app.workers import RealWorker
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, write_manifest
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY


def _setup(tmp_path: Path) -> tuple[RealWorker, Any, Path]:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "licenses").symlink_to(project_root() / "configs" / "licenses")  # the real, approved texts
    w = RealWorker(root)
    entry = w.catalog.get("qwen3.5-0.8b")
    assert entry is not None
    snap = root / ".cache" / "hf" / "hub" / f"models--{entry.repo.replace('/', '--')}" / "snapshots" / entry.revision
    snap.mkdir(parents=True)
    (snap / "LICENSE").write_text("upstream licence\n")
    run = root / "outputs" / "graph" / "runs" / "merge-1"
    model = run / "merged"
    model.mkdir(parents=True)
    write_manifest(run, RunManifest(run_id="merge-1", name="m", stage="train", task="merge_lora",
                                    base_model=entry.repo, base_model_revision=entry.revision,
                                    parent_run_id="g-20261013-101500-abcdef"))  # fmt: skip
    (model / "config.json").write_text(json.dumps({"_name_or_path": "/workspace/.cache/hf/x", "model_type": "qwen3_5"}))
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}")
    (model / "merge_report.json").write_text("{}")  # internal: never exported
    return w, entry, model


def _job(model: Path, run_dir: Path, dest: str = "outputs/exports/my-model") -> NodeJob:
    spec = REGISTRY.get("export.hf_dir")
    assert spec is not None
    ref = ArtifactRef("Checkpoint", str(model), "merge-1:abc")
    return NodeJob("x", spec, spec.validate_params({"dest": dest}), {"model": ref}, run_dir)


def test_native_export_is_portable_licensed_and_atomic(tmp_path: Path) -> None:
    w, entry, model = _setup(tmp_path)
    out = w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-1"))
    dest = w.root / "outputs" / "exports" / "my-model"
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["ATTRIBUTION.json", "LICENSE", "LICENSE.upstream", "NOTICE", "README.md", "config.json",
                     "model.safetensors", "tokenizer.json"]  # fmt: skip
    assert json.loads((dest / "config.json").read_text())["_name_or_path"] == f"{entry.repo}@{entry.revision}"
    assert "Apache" in (dest / "LICENSE").read_text()
    assert not list(dest.parent.glob(".my-model.tmp-*"))
    receipt = json.loads(Path(out.outputs["artifact"].path).read_text())
    assert receipt["dest"] == "outputs/exports/my-model"
    assert "model.safetensors" in receipt["files"]
    again = w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-2"))  # never mixed into an existing export
    assert json.loads(Path(again.outputs["artifact"].path).read_text())["dest"].startswith("outputs/exports/my-model-")


def test_remote_code_in_the_folder_is_refused_whatever_the_catalog_says(tmp_path: Path) -> None:
    w, _, model = _setup(tmp_path)
    (model / "modeling_x.py").write_text("print('hi')\n")
    with pytest.raises(export_worker.ExportUnsupportedError):
        w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-3"))
    assert not (w.root / "outputs" / "exports").exists() or not any((w.root / "outputs" / "exports").iterdir())


@pytest.mark.parametrize("dest", ["src/evil", "outputs/graph/runs/x", "outputs", "outputs/system/x", ".cache/x"])
def test_destination_must_be_a_folder_of_its_own_under_outputs(tmp_path: Path, dest: str) -> None:
    w, _, model = _setup(tmp_path)
    with pytest.raises(export_worker.ExportDestinationError):
        w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-4", dest=dest))


def test_extra_json_with_auto_map_or_a_machine_path_is_refused(tmp_path: Path) -> None:
    w, _, model = _setup(tmp_path)
    (model / "extra.json").write_text(json.dumps({"auto_map": {"AutoModel": "m.M"}}))
    with pytest.raises(export_worker.ExportUnsupportedError):
        w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-5"))
    (model / "extra.json").unlink()
    (model / "notes.txt").write_text("from /home/someone/data\n")
    with pytest.raises(export_worker.ExportUnsupportedError):
        w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-6"))


def test_a_token_inside_the_export_is_never_written(tmp_path: Path) -> None:
    w, _, model = _setup(tmp_path)
    fake = "hf_" + "Ab" * 17  # token SHAPE built at run time (never a literal in the repo)
    (model / "tokenizer_config.json").write_text(json.dumps({"note": fake}))
    with pytest.raises(export_worker.ExportSecretError) as ei:
        w.run(_job(model, w.root / "outputs" / "graph" / "runs" / "x-7"))
    assert fake not in str(ei.value)
    assert ei.value.params == {"node": "x", "files": "tokenizer_config.json"}
    assert str(ei.value) == "credential-shaped string (HF_TOKEN) in the export"
    exports = w.root / "outputs" / "exports"
    assert not exports.exists() or not any(exports.iterdir())  # temp folder removed, nothing published
