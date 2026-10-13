"""P016 §3: every bad engine option of a train node is its own CFG-005 issue, with its dotted key and rule."""

from __future__ import annotations

from typing import Any

from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY


def _wf(advanced: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {"id": "m", "type": "model.hf_pick", "version": "1.0.0", "params": {"model": "qwen3.5-0.8b"}},
        {"id": "d", "type": "data.hf_load", "version": "1.0.0", "params": {}},
        {"id": "t", "type": "train.lora_vlm", "version": "1.0.0", "params": {"advanced": advanced}},
    ]
    edges = [{"from": ["m", "model"], "to": ["t", "base_model"]}, {"from": ["d", "dataset"], "to": ["t", "train_data"]}]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


def test_each_bad_engine_option_is_reported_under_its_own_key() -> None:
    bad = {"dataloader.num_workers": "four", "dataloader.drop_last": 3, "model._target_": "os.system"}
    issues = [i for i in validate(_wf(bad), REGISTRY) if i.code == "CFG-005"]
    found = {(i.params["key"], i.params["rule"]) for i in issues}
    assert found == {
        ("dataloader.num_workers", "type"),
        ("dataloader.drop_last", "type"),
        ("model._target_", "not_offered"),
    }
    assert all(i.param == "advanced" for i in issues)
    assert all(i.node_id == "t" for i in issues)
    assert not any("os.system" in str(i.params) or "four" in str(i.params) for i in issues)  # values never echoed


def test_the_error_type_literal_matches_the_train_constant() -> None:
    """graph never imports train: its literal copy must not drift from train.advanced.ERROR_TYPE."""
    from vlm_pipeline.graph.validation import ADVANCED_ERROR_TYPE
    from vlm_pipeline.train.advanced import ERROR_TYPE

    assert ERROR_TYPE == ADVANCED_ERROR_TYPE
