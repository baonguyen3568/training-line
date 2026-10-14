"""First-party groups in GET /api/node-types (P4-4): ports and params derived from the inner nodes.

The golden files in schemas/group_examples/ are shared with the frontend, which derives the same shape for user
groups (frontend test reads them too). Regenerate the expected parts with VLM_UPDATE_GOLDEN=1 after a deliberate
change and review the diff.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.paths import project_root
from vlm_pipeline.graph.groups import GroupShapeError, group_to_api
from vlm_pipeline.graph.workflow import GroupDefinition
from vlm_pipeline.nodes import REGISTRY

EXAMPLES = sorted((project_root() / "schemas" / "group_examples").glob("*.json"))


def _derive(example: dict[str, Any]) -> dict[str, Any]:
    groups = {g["id"]: GroupDefinition.model_validate(g) for g in [*example["uses"], example["group"]]}
    try:
        return group_to_api(groups[example["group"]["id"]], REGISTRY, groups)
    except GroupShapeError as exc:
        return {"error": str(exc)}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_golden_group_shapes(path: Path) -> None:
    example = json.loads(path.read_text(encoding="utf-8"))
    derived = _derive(example)
    if os.environ.get("VLM_UPDATE_GOLDEN") == "1":
        path.write_text(json.dumps({**example, "expected": derived}, indent=1, ensure_ascii=False) + "\n")
    assert derived == example.get("expected"), f"{path.name}: rerun with VLM_UPDATE_GOLDEN=1 if the change is wanted"


def test_the_examples_cover_a_plain_a_nested_and_a_broken_group() -> None:
    by_name = {p.stem: json.loads(p.read_text(encoding="utf-8"))["expected"] for p in EXAMPLES}
    lora, nested = by_name["lora"], by_name["nested"]
    assert lora["inputs"]["model"] == {"type": "Model", "optional": False, "many": False}  # two inner ports, one type
    assert lora["params_schema"]["properties"]["val_fraction"]["default"] == 0.2  # the group's own value
    assert (lora["category"], lora["runner"]) == ("train", "container")  # lane of the output's node (merge)
    assert nested["category"] == "eval"  # its output comes from eval.benchmarks, not from the nested group
    assert nested["params_schema"]["properties"]["epochs"]["default"] == 2  # set by the outer group
    assert nested["outputs"]["report"]["type"] == "EvalReport"
    assert "different types" in by_name["mixed_types"]["error"]


def test_a_group_with_a_step_that_cannot_run_is_unavailable() -> None:
    example = json.loads((project_root() / "schemas" / "group_examples" / "lora.json").read_text(encoding="utf-8"))
    group = GroupDefinition.model_validate(example["group"])
    item = group_to_api(group, REGISTRY, {group.id: group}, can_run=lambda spec: spec.type_id != "train.merge_lora")
    assert (item["available"], item["unavailable_reason"]) == (
        False,
        {"code": "CFG-011", "inner_type": "train.merge_lora"},
    )


def test_every_shipped_group_derives_and_has_its_text() -> None:
    """configs/groups/*.json ship with the app: each loads, derives and has vi + en title/desc (else the palette
    shows a raw key or loses the group)."""
    from vlm_pipeline.graph.groups import GroupShapeError, first_party_groups, group_to_api
    from vlm_pipeline.i18n import load

    groups = first_party_groups()
    texts = [load("vi"), load("en")]
    for group in groups.values():
        try:
            item = group_to_api(group, REGISTRY, groups)
        except GroupShapeError as exc:
            pytest.fail(f"{group.id}: {exc}")
        assert item["category"] != "group"
        missing = [k for k in (item["title_key"], item["desc_key"]) for t in texts if k not in t]
        assert missing == [], f"{group.id}: add i18n {sorted(set(missing))}"
