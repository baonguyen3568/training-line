"""CFG-010: the fine-tune node must be the one the model catalog vetted for the picked model (``train_node``)."""

from __future__ import annotations

from typing import Any

import pytest

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY

PARSE = "nemotron-parse-v1.1"  # train_node: train.sft_vlm
QWEN = "qwen3.5-0.8b"  # train_node: train.lora_vlm


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def _wf(model: str | None, train: str, *extra_nodes: dict[str, Any], edges: tuple[dict[str, Any], ...] = ()) -> Any:
    pick = _n("m", "model.hf_pick") if model is None else _n("m", "model.hf_pick", model=model)
    nodes = [pick, _n("d", "data.hf_load"), _n("t", train), *extra_nodes]
    wires = [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data"), *edges]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": wires}


def _cfg010(workflow: Any) -> list[Any]:
    return [i for i in validate(workflow, REGISTRY) if i.code == "CFG-010"]


def test_test_models_have_the_assumed_train_nodes() -> None:
    catalog = load_catalog()
    assert catalog.get(PARSE).train_node == "train.sft_vlm"  # type: ignore[union-attr]
    assert catalog.get(QWEN).train_node == "train.lora_vlm"  # type: ignore[union-attr]


@pytest.mark.parametrize(("model", "train"), [(PARSE, "train.sft_vlm"), (QWEN, "train.lora_vlm")])
def test_vetted_pair_passes(model: str, train: str) -> None:
    assert _cfg010(_wf(model, train)) == []


@pytest.mark.parametrize(("model", "train"), [(PARSE, "train.lora_vlm"), (QWEN, "train.sft_vlm")])
def test_other_train_node_is_blocked_with_the_supported_one(model: str, train: str) -> None:
    [issue] = _cfg010(_wf(model, train))
    entry = load_catalog().get(model)
    assert entry is not None
    assert (issue.severity, issue.node_id, issue.port) == ("error", "t", "base_model")
    assert dict(issue.params) == {"node": "t", "model": entry.display_name, "supported": entry.train_node}


def test_default_model_counts_when_param_is_omitted() -> None:
    default = load_catalog().default
    wrong = "train.lora_vlm" if default.train_node == "train.sft_vlm" else "train.sft_vlm"
    assert len(_cfg010(_wf(None, wrong))) == 1


def test_followed_through_a_previous_fine_tune() -> None:
    # Parse -> full fine-tune -> checkpoint -> LoRA: still Parse underneath.
    sft = _n("s", "train.sft_vlm")
    edges = (_e("m", "model", "s", "base_model"), _e("d", "dataset", "s", "train_data"))
    wf = _wf(PARSE, "train.lora_vlm", sft, edges=edges)
    wf["edges"][0] = _e("s", "model", "t", "base_model")
    assert [i.node_id for i in _cfg010(wf)] == ["t"]


def test_unknown_origin_is_not_guessed() -> None:
    # Invalid model param (already CFG-005) or unwired base_model (CFG-003): no CFG-010 on top.
    assert _cfg010(_wf("no-such-model", "train.lora_vlm")) == []
    wf = _wf(PARSE, "train.lora_vlm")
    wf["edges"] = wf["edges"][1:]
    assert _cfg010(wf) == []


def test_every_shipped_template_passes() -> None:
    from vlm_pipeline.app.templates import load_templates
    from vlm_pipeline.common.paths import project_root

    for template in load_templates(project_root()):
        assert _cfg010(template["workflow"]) == [], template["id"]


@pytest.mark.parametrize("locale", ["vi", "en"])
def test_issue_renders_without_raw_ids(locale: str) -> None:
    from dataclasses import asdict

    from vlm_pipeline.app.main import _render_issue

    [issue] = _cfg010(_wf(PARSE, "train.lora_vlm"))
    out = _render_issue(asdict(issue), locale)
    assert "{" not in out["title"] + out["what"]
    assert load_catalog().get(PARSE).display_name in out["what"]  # type: ignore[union-attr]
