"""data.hf_load with a user-added dataset (ADR-021 H5): the id comes from the hidden repo + revision, the catalog's
facts about the default repo never apply to it, and downstream it counts as local data."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from vlm_pipeline.download.resolver import user_dataset_id
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY
from vlm_pipeline.nodes.source import HubLoadParams

REV = "a" * 40
USER = {"user_dataset": user_dataset_id("Org/My_Data", REV, "default", "train"), "user_repo": "Org/My_Data",
        "user_revision": REV}  # fmt: skip


def test_the_id_must_come_from_the_hidden_repo_and_revision() -> None:
    assert HubLoadParams(**USER).user_dataset == USER["user_dataset"]
    for bad in (
        {**USER, "user_repo": "Org/Other"},
        {**USER, "user_revision": "b" * 40},
        {"user_dataset": "user--ds--x"},
    ):
        with pytest.raises(ValidationError):
            HubLoadParams(**bad)


@pytest.mark.parametrize("bad", ["user--ds--..", "user--ds--a/b", "user--ds--A", "user--x"])
def test_ids_are_path_safe(bad: str) -> None:
    with pytest.raises(ValidationError):
        HubLoadParams(user_dataset=bad)


def _wf(**load: Any) -> dict[str, Any]:
    nodes = [
        {"id": "m", "type": "model.hf_pick", "version": "1.0.0", "params": {"model": "qwen3.5-0.8b"}},
        {"id": "d", "type": "data.hf_load", "version": "1.0.0", "params": load},
        {"id": "t", "type": "train.lora_vlm", "version": "1.0.0", "params": {}},
    ]
    edges = [{"from": ["m", "model"], "to": ["t", "base_model"]}, {"from": ["d", "dataset"], "to": ["t", "train_data"]}]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


def test_catalog_facts_of_the_default_repo_never_apply_to_a_user_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eval-only catalog repo would be an error (TRAIN-HELDOUT-DATA); the user's own dataset is not that repo."""
    from vlm_pipeline.common.catalog import DatasetEntry, load_dataset_catalog
    from vlm_pipeline.nodes import standards

    cord = load_dataset_catalog().datasets[0]
    eval_only = DatasetEntry.model_validate({**cord.model_dump(), "use": "eval_only", "heldout_splits": cord.splits})
    only_eval = SimpleNamespace(datasets=(eval_only,))
    monkeypatch.setattr(standards, "load_dataset_catalog", lambda: only_eval)

    def rules(wf: dict[str, Any]) -> set[object]:
        return {i.params.get("rule") for i in validate(wf, REGISTRY) if i.code == "STD-001"}

    assert "TRAIN-HELDOUT-DATA" in rules(_wf(max_samples=100))
    assert "TRAIN-HELDOUT-DATA" not in rules(_wf(max_samples=100, **USER))


def test_a_user_dataset_counts_as_local_data_downstream() -> None:
    from vlm_pipeline.nodes.rules import input_kinds
    from vlm_pipeline.nodes.spec import GraphView

    specs = {"d": REGISTRY.get("data.hf_load"), "t": REGISTRY.get("train.lora_vlm")}
    params = {"d": HubLoadParams(**USER)}
    view = GraphView(specs=specs, params=params, sources={("t", "train_data"): ("d",)})  # type: ignore[arg-type]
    assert input_kinds(view, "t")["train_data"] == "dataset:local"
    view_hub = GraphView(specs=specs, params={"d": HubLoadParams()}, sources={("t", "train_data"): ("d",)})  # type: ignore[arg-type]
    assert input_kinds(view_hub, "t")["train_data"] == "dataset:hub"


def test_a_user_dataset_not_added_or_not_converted_is_reported_before_run(tmp_path: Any) -> None:
    """validate == worker: never added here -> DATA-013 (offer to add it); added but not converted -> NET-002 warn."""
    from vlm_pipeline.common import catalog

    def issues(status: dict[str, Any] | None) -> list[tuple[str, str, str | None]]:
        catalog.set_user_dataset_lookup(lambda _id: status)
        try:
            found = validate(_wf(**USER), REGISTRY, workspace=tmp_path)
        finally:
            catalog.set_user_dataset_lookup(None)
        return [(i.code, i.severity, i.param) for i in found if i.node_id == "d"]

    assert issues(None) == [("DATA-013", "error", "user_dataset")]
    missing = {"repo": "Org/My_Data", "kind": "dataset", "catalog_id": USER["user_dataset"]}
    assert issues({"rows": None, "download": missing}) == [("NET-002", "warn", "user_dataset")]
    assert issues({"rows": 500, "download": None}) == []
