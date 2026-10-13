"""G-3: /api/graphs/validate returns resolved quick fixes next to action_ids, from the node's validated params."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from vlm_pipeline.app.main import create_app
from vlm_pipeline.app.services import DefaultServices
from vlm_pipeline.nodes import REGISTRY


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def test_every_issue_carries_actions_matching_action_ids(tmp_path: Path) -> None:
    workflow = {"format": "vlm-workflow", "schema_version": 1, "name": "t",
                "nodes": [_n("m", "model.hf_pick", model="nemotron-parse-v1.1"), _n("d", "data.hf_load"),
                          _n("t", "train.lora_vlm")],
                "edges": [{"from": ["m", "model"], "to": ["t", "base_model"]},
                          {"from": ["d", "dataset"], "to": ["t", "train_data"]}]}  # fmt: skip
    client = TestClient(create_app(DefaultServices(registry=REGISTRY, root=tmp_path), secure=False))
    errors = client.post("/api/graphs/validate", json=workflow).json()["errors"]
    cfg010 = next(e for e in errors if e["code"] == "CFG-010")
    assert cfg010["action_ids"] == [a["id"] for a in cfg010["actions"]]
    assert "pick_other_asset" in cfg010["action_ids"]
    assert all(a.get("node_id") == "t" for a in cfg010["actions"])
    assert all("action_ids" in e and "actions" in e for e in errors)


def test_advanced_options_are_listed_per_train_node_and_model(tmp_path: Path) -> None:
    """P016 §3 (c): the panel reads the engine options lazily; a model narrows them to its own template."""
    client = TestClient(create_app(DefaultServices(registry=REGISTRY, root=tmp_path), secure=False))
    every = client.get("/api/node-types/train.lora_vlm/advanced").json()["options"]
    assert every
    assert {"key", "group", "type", "default", "choices", "min", "max", "templates"} <= set(every[0])
    assert all(o["group"] == o["key"].rsplit(".", 1)[0] for o in every)
    parse = client.get("/api/node-types/train.lora_vlm/advanced", params={"model": "nemotron-parse-v1.1"}).json()
    assert 0 < len(parse["options"]) <= len(every)
    assert all("nemotron-parse-v1.1" in o["templates"] for o in parse["options"])
    assert client.get("/api/node-types/data.filter/advanced").status_code == 404  # no advanced param
    assert client.get("/api/node-types/nope/advanced").status_code == 404
