"""M1b B-0: nodes the real worker cannot run yet (CFG-011) and non-portable exports (EXP-002) fail at validate time."""

from __future__ import annotations

from typing import Any

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.graph import validate
from vlm_pipeline.graph.worker import worker_supports
from vlm_pipeline.nodes import REGISTRY

PARSE = "nemotron-parse-v1.1"  # remote_code: pinned_prepared
QWEN = "qwen3.5-0.8b"  # remote_code: none


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _wf(model: str, *, train: str, export: bool) -> dict[str, Any]:
    nodes = [_n("m", "model.hf_pick", model=model), _n("d", "data.hf_load"), _n("t", train)]
    edges = [{"from": ["m", "model"], "to": ["t", "base_model"]}, {"from": ["d", "dataset"], "to": ["t", "train_data"]}]
    if export:
        out = "model" if train == "train.sft_vlm" else None
        src = ("t", out) if out else ("m", "model")
        nodes.append(_n("x", "export.hf_dir"))
        edges.append({"from": list(src), "to": ["x", "model"]})
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


class _Only:
    def __init__(self, *types: str) -> None:
        self.types = set(types)

    def supports(self, spec: Any) -> bool:
        return spec.type_id in self.types


def _codes(workflow: dict[str, Any], **kw: Any) -> list[tuple[str, str | None]]:
    return [(i.code, i.node_id) for i in validate(workflow, REGISTRY, **kw)]


def test_nodes_the_worker_cannot_run_are_reported_before_the_run() -> None:
    worker = _Only("model.hf_pick", "data.hf_load", "train.sft_vlm")
    issues = validate(
        _wf(PARSE, train="train.sft_vlm", export=True), REGISTRY, can_run=lambda s: worker_supports(worker, s)
    )
    cfg011 = [i for i in issues if i.code == "CFG-011"]
    assert [(i.node_id, dict(i.params)) for i in cfg011] == [("x", {"node": "x", "node_type": "export.hf_dir"})]


def test_workers_without_supports_run_everything() -> None:
    spec = REGISTRY.get("export.hf_dir")
    assert spec is not None
    assert worker_supports(object(), spec) is True  # e.g. FakeWorker in demo mode
    assert ("CFG-011", "x") not in _codes(_wf(QWEN, train="train.lora_vlm", export=True))


def test_export_of_a_remote_code_model_is_blocked_on_the_canvas() -> None:
    entry = load_catalog().get(PARSE)
    assert entry is not None
    assert entry.remote_code != "none"
    [issue] = [i for i in validate(_wf(PARSE, train="train.sft_vlm", export=True), REGISTRY) if i.code == "EXP-002"]
    assert (issue.node_id, issue.port, dict(issue.params)) == ("x", "model", {"node": "x", "model": entry.display_name})


def test_export_of_a_native_model_passes_the_rule() -> None:
    assert ("EXP-002", "x") not in _codes(_wf(QWEN, train="train.lora_vlm", export=True))


def test_real_worker_supports_only_what_it_runs(tmp_path: Any) -> None:
    from vlm_pipeline.app.workers import RealWorker

    w = RealWorker(tmp_path)
    assert w.supports(REGISTRY.get("train.sft_vlm"))  # type: ignore[arg-type]
    assert not w.supports(REGISTRY.get("export.hf_dir"))  # type: ignore[arg-type]
