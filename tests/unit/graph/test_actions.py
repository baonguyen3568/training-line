"""G-3: validate issues and error events carry resolved quick fixes computed from the node's own params."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vlm_pipeline.graph.actions import issue_actions
from vlm_pipeline.graph.errors import UserError


class _TrainParams(BaseModel):
    max_seq_len: int = Field(2048, ge=256, le=8192)
    skip_invalid_samples: bool = False


def test_patch_values_are_final_and_both_lists_agree() -> None:
    err = UserError("DATA-001", {"idx": 3, "image_count": 1, "slot_count": 0})
    out = issue_actions(err, node_id="n4", node_params=_TrainParams().model_dump(), params_model=_TrainParams)
    assert [a["id"] for a in out["actions"]] == out["action_ids"]
    by_id: dict[str, Any] = {a["id"]: a for a in out["actions"]}
    assert by_id["increase_max_length"]["patch"] == [{"param": "max_seq_len", "from": 2048, "value": 4096}]
    assert by_id["skip_bad_samples"]["patch"] == [{"param": "skip_invalid_samples", "from": False, "value": True}]
    assert all(a["node_id"] == "n4" for a in out["actions"])


def test_a_patch_that_would_change_nothing_is_not_offered() -> None:
    at_max = _TrainParams(max_seq_len=8192, skip_invalid_samples=True).model_dump()
    out = issue_actions(UserError("DATA-001", {}), node_id="n4", node_params=at_max, params_model=_TrainParams)
    assert "increase_max_length" not in out["action_ids"]
    assert "skip_bad_samples" not in out["action_ids"]


def test_without_the_node_params_no_patch_is_guessed() -> None:
    out = issue_actions(UserError("DATA-001", {}))
    assert all(a["kind"] != "patch" or a["patch"] is None for a in out["actions"])
    assert "increase_max_length" not in out["action_ids"]
    assert out["action_ids"] == [a["id"] for a in out["actions"]]
