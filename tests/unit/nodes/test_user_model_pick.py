"""ADR-021 H3: model.hf_pick picks a user-added model (user_model wins) and the one lookup follows it everywhere."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vlm_pipeline.common.catalog import CatalogEntry, set_user_model_lookup
from vlm_pipeline.download import resolver
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.source import ModelPickParams, picked_model_id

SHA = "b" * 40
USER_ID = resolver.user_model_id("Org/My_Model", SHA)


def _entry() -> CatalogEntry:
    family = resolver.load_families().for_architecture("Qwen3_5ForConditionalGeneration")
    assert family is not None
    return resolver.user_entry(repo="Org/My_Model", revision=SHA, family=family, params_b=2.0, disk_gb=4.0,
                               license_id="apache-2.0", gated=False)  # fmt: skip


@pytest.fixture
def added() -> Any:
    entry = _entry()
    set_user_model_lookup(lambda model_id: entry if model_id == entry.id else None)
    yield entry
    set_user_model_lookup(None)


def _pick(**params: Any) -> dict[str, Any]:
    return {"id": "m", "type": "model.hf_pick", "version": "1.0.0", "params": params}


def _user_params() -> dict[str, Any]:
    return {"user_model": USER_ID, "user_repo": "Org/My_Model", "user_revision": SHA}


def test_user_model_wins_over_the_catalog_choice() -> None:
    params = ModelPickParams(**_user_params())
    assert picked_model_id(params) == USER_ID
    assert picked_model_id(ModelPickParams()) == ModelPickParams().model  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "params",
    [
        {"user_model": "user--.."},
        {"user_model": "user--a/b"},
        {"user_model": "user--A"},
        {"user_model": USER_ID},  # without the repo + revision it was made from
        {**_user_params(), "user_repo": "Org/Other"},
        {**_user_params(), "user_revision": "c" * 40},
        {"user_repo": "Org/My_Model", "user_revision": SHA},  # a descriptor without the id
    ],
)
def test_a_user_model_id_is_path_safe_and_matches_its_descriptor(params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelPickParams(**params)


def _wf(*nodes: dict[str, Any], edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": list(nodes), "edges": edges or []}


def test_a_workflow_naming_a_model_not_added_here_offers_to_add_it(tmp_path: Path) -> None:
    issues = [i for i in validate(_wf(_pick(**_user_params())), REGISTRY, workspace=tmp_path) if i.code == "MODEL-006"]
    [issue] = issues
    assert (issue.node_id, issue.param) == ("m", "user_model")
    assert issue.params == {"node": "m", "repo": "Org/My_Model", "revision": SHA}


def test_an_added_user_model_flows_to_the_train_node_checks(tmp_path: Path, added: CatalogEntry) -> None:
    train = {"id": "t", "type": "train.sft_vlm", "version": "1.0.0", "params": {}}
    data = {"id": "d", "type": "data.hf_load", "version": "1.0.0", "params": {}}
    edges = [{"from": ["m", "model"], "to": ["t", "base_model"]}, {"from": ["d", "dataset"], "to": ["t", "data"]}]
    codes = {i.code for i in validate(_wf(_pick(**_user_params()), train, data, edges=edges), REGISTRY,
                                      workspace=tmp_path)}  # fmt: skip
    assert "MODEL-006" not in codes
    assert "CFG-010" in codes  # full SFT is catalog-only: a user model offers LoRA (the same rule as the catalog)
