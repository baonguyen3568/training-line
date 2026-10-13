"""P018 §3.1: validate warns NET-002 for a picked model / hub dataset not on this machine (same check as the run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_pipeline.common.catalog import load_catalog, load_dataset_catalog
from vlm_pipeline.common.local_cache import hub_snapshot, prepared_model_dir
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY


def _wf(*nodes: dict[str, Any]) -> dict[str, Any]:
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": list(nodes), "edges": []}


def _pick(model: str) -> dict[str, Any]:
    return {"id": "m", "type": "model.hf_pick", "version": "1.0.0", "params": {"model": model}}


def _hub(repo: str) -> dict[str, Any]:
    return {"id": "d", "type": "data.hf_load", "version": "1.0.0", "params": {"repo": repo}}


def _net002(root: Path | None, *nodes: dict[str, Any]) -> list[Any]:
    return [i for i in validate(_wf(*nodes), REGISTRY, workspace=root) if i.code == "NET-002"]


def test_missing_assets_are_warned_with_the_download_params(tmp_path: Path) -> None:
    model = load_catalog().get("qwen3.5-0.8b")
    dataset = next(d for d in load_dataset_catalog().datasets if d.id == "cord-v2")
    assert model is not None
    issues = _net002(tmp_path, _pick(model.id), _hub(dataset.repo))
    by_node = {i.node_id: i for i in issues}
    assert {i.severity for i in issues} == {"warn"}  # Run stays enabled: the UI offers "download all missing"
    assert by_node["m"].params == {"repo": model.repo, "reason": "not_cached", "catalog_id": model.id,
                                   "kind": "model", "size_gb": model.disk_gb, "name": model.display_name,
                                   "license": model.license}  # fmt: skip
    assert (by_node["d"].params["name"], by_node["d"].params["license"]) == (dataset.display_name, dataset.license)
    assert by_node["m"].param == "model"
    assert by_node["d"].params["catalog_id"] == dataset.id
    assert by_node["d"].params["kind"] == "dataset"
    hub_snapshot(tmp_path, "model", model.repo, model.revision).mkdir(parents=True)
    hub_snapshot(tmp_path, "dataset", dataset.repo, dataset.revision).mkdir(parents=True)
    assert _net002(tmp_path, _pick(model.id), _hub(dataset.repo)) == []


def test_a_remote_code_model_needs_its_prepared_copy(tmp_path: Path) -> None:
    [warn] = _net002(tmp_path, _pick("nemotron-parse-v1.1"))
    assert warn.params["reason"] == "not_prepared"
    prepared = prepared_model_dir(tmp_path, "nemotron-parse-v1.1")
    assert prepared is not None
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    assert _net002(tmp_path, _pick("nemotron-parse-v1.1")) == []


def test_no_workspace_no_warning() -> None:
    assert _net002(None, _pick("qwen3.5-0.8b")) == []
