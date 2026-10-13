"""Curation data nodes (Release 1.0, stream B): params rules and how the standards see through them."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.data import DATA_FILTER, DATA_MIX, PASS_THROUGH
from vlm_pipeline.nodes.spec import Registry
from vlm_pipeline.nodes.standards import STANDARDS_DIR, load_standards, standards_rule


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def test_filter_answer_length_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="min_target_chars"):
        DATA_FILTER.validate_params({"min_target_chars": 500, "max_target_chars": 500})


def test_mix_needs_two_datasets() -> None:
    required = sorted(name for name, port in DATA_MIX.inputs.items() if not port.optional)
    assert required == ["dataset_1", "dataset_2"]


def _lora(*between: str) -> dict[str, Any]:
    """Catalog dataset (100 samples) -> curation nodes -> LoRA training."""
    chain = ["d", *(f"c{i}" for i in range(len(between))), "t"]
    nodes = [_n("m", "model.hf_pick", model="qwen3.5-0.8b"), _n("d", "data.hf_load", max_samples=100)]
    nodes += [_n(f"c{i}", type_id) for i, type_id in enumerate(between)] + [_n("t", "train.lora_vlm")]
    edges = [_e("m", "model", "t", "base_model")]
    types = dict(zip(chain[1:-1], between, strict=True))
    for a, b in itertools.pairwise(chain):
        ports = ["train_data"] if b == "t" else ["dataset_1", "dataset_2"] if types[b] == "data.mix" else ["dataset"]
        edges += [_e(a, "dataset", b, port) for port in ports]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


def _small_data(tmp_path: Path, wf: dict[str, Any]) -> list[str]:
    """Node ids flagged by a test rule "training data has fewer than 200 samples" (catalog count, max_samples)."""
    rule = {
        "id": "TEST-SMALL",
        "applies_to": {"node": "train.lora_vlm", "port": "train_data"},
        "check": {"kind": "sample_count", "below": 200},
        "level": "warn",
        "message": "standards.train.too_few_samples",
        "source": "test",
    }
    (tmp_path / STANDARDS_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_path / STANDARDS_DIR / "t.yaml").write_text(yaml.safe_dump({"schema_version": 1, "rules": [rule]}))
    specs = REGISTRY.all()
    registry = Registry(specs, rules=(standards_rule(load_standards(tmp_path, specs)),), allow_unpinned=True)
    return [str(i.node_id) for i in validate(wf, registry) if i.code == "STD-001"]


def test_standards_follow_the_dataset_through_curation_nodes(tmp_path: Path) -> None:
    """Filter / dedupe / convert keep the dataset's identity: the catalog checks still see it behind them."""
    assert _small_data(tmp_path, _lora()) == ["t"]  # 100 catalog samples
    assert _small_data(tmp_path, _lora(*sorted(PASS_THROUGH))) == ["t"]
    assert _small_data(tmp_path, _lora("data.mix")) == []  # unknown size after a mix: never guessed
