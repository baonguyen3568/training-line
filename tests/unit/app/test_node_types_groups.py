"""GET /api/node-types lists first-party groups (contract .team/proposals/016b) with rendered titles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.graph import groups as groups_module
from vlm_pipeline.graph.workflow import GroupDefinition
from vlm_pipeline.nodes import REGISTRY


def test_groups_are_listed_next_to_node_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = json.loads((project_root() / "schemas" / "group_examples" / "lora.json").read_text(encoding="utf-8"))
    group = GroupDefinition.model_validate(example["group"])
    monkeypatch.setattr(groups_module, "first_party_groups", lambda root=None: {group.id: group})
    client = TestClient(create_app(DefaultServices(registry=REGISTRY, root=tmp_path), secure=False))
    body = client.get("/api/node-types").json()
    [item] = body["groups"]
    assert item["group_id"] == "group.example_lora"
    assert item["title"]  # rendered (falls back to the key while research writes the copy)
    assert not any(n["type_id"] == "group.example_lora" for n in body["node_types"])  # node_types stay runnable types
    assert item["inputs"] == example["expected"]["inputs"]
