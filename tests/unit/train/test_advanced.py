"""Advanced engine options of the train nodes (P016 §3): offered keys only, typed, bounded, per model template."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.advanced import check_advanced, keys_not_for, set_dotted
from vlm_pipeline.train.config import LoraParams
from vlm_pipeline.train.recipe import build_recipe, load_template, template_path


def test_offered_values_pass() -> None:
    values = {"optimizer.eps": 1e-8, "model.backend.attn": "flex", "dataloader.num_workers": 2, "rng.ranked": True}
    assert check_advanced(values) == values
    assert LoraParams(advanced=values).advanced == values


@pytest.mark.parametrize(
    ("values", "needle"),
    [
        ({"model._target_": "os.system"}, "not an offered"),  # denied keys never reach the file
        ({"optimizer.lr": 1.0}, "not an offered"),  # the node's own field
        ({"dataloader.num_workers": 512}, "range"),  # shared-host DoS
        ({"dataloader.num_workers": True}, "type"),  # bool is not an int here
        ({"rng.ranked": 1}, "type"),
        ({"model.backend.attn": "flash3"}, "choice"),
        ({"model.backend.attn": "${oc.env:HF_TOKEN}"}, "resolver"),
        ({f"k{i}": 1 for i in range(101)}, "at most"),
    ],
)
def test_bad_values_are_refused(values: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        check_advanced(values)
    with pytest.raises(ValidationError):
        LoraParams(advanced=values)


def test_keys_follow_the_model_template() -> None:
    # TE FusedAdam options exist only in the Nemotron-Parse templates (full fine-tune); Qwen trains with torch AdamW.
    assert keys_not_for("qwen3.5-0.8b", {"optimizer.adam_w_mode": True, "optimizer.eps": 1e-8}) == [
        "optimizer.adam_w_mode"
    ]
    assert keys_not_for("nemotron-parse-v1.1", {"optimizer.adam_w_mode": True}) == []


def test_set_dotted_creates_only_what_is_missing() -> None:
    recipe: dict[str, Any] = {"model": {"backend": {"attn": "sdpa", "linear": "torch"}}}
    set_dotted(recipe, "model.backend.attn", "flex")
    set_dotted(recipe, "dataloader.num_workers", 2)
    assert recipe == {"model": {"backend": {"attn": "flex", "linear": "torch"}}, "dataloader": {"num_workers": 2}}


def test_recipe_applies_advanced_after_the_template_and_before_the_node() -> None:
    entry = load_catalog().get("qwen3.5-0.8b")
    assert entry is not None
    template = load_template(template_path(project_root(), entry))
    params = LoraParams(advanced={"model.backend.attn": "flex", "dataloader.num_workers": 2})
    recipe = build_recipe(template, params, model_dir="/m", dataset_dir="/d", run_dir="/r")
    assert recipe["model"]["backend"]["attn"] == "flex"
    assert recipe["dataloader"]["num_workers"] == 2
    assert recipe["optimizer"]["lr"] == params.learning_rate  # the node's own fields are set last
    assert LoraParams().advanced == {}
    assert (
        build_recipe(template, LoraParams(), model_dir="/m", dataset_dir="/d", run_dir="/r")["model"]["backend"]["attn"]
        == template["model"]["backend"]["attn"]
    )  # empty advanced changes nothing


def test_train_support_check_refuses_a_key_the_model_template_lacks() -> None:
    from vlm_pipeline.app.workers import _check_train
    from vlm_pipeline.graph.worker import SupportQuery
    from vlm_pipeline.nodes import REGISTRY
    from vlm_pipeline.train.advanced import AdvancedNotForModelError

    spec = REGISTRY.get("train.lora_vlm")
    model = load_catalog().get("qwen3.5-0.8b")
    assert spec is not None
    assert model is not None
    ok = LoraParams(advanced={"optimizer.eps": 1e-8})
    _check_train(SupportQuery("t", spec, ok, model, {}))
    wrong = LoraParams(advanced={"optimizer.adam_w_mode": True})
    with pytest.raises(AdvancedNotForModelError) as info:
        _check_train(SupportQuery("t", spec, wrong, model, {}))
    assert (info.value.code, info.value.params["value"], info.value.params["rule"]) == (
        "CFG-005",
        "optimizer.adam_w_mode",
        "not_for_model",
    )


def test_set_dotted_never_overwrites_a_non_section() -> None:
    recipe: dict[str, Any] = {"dataloader": {"collate_fn": "pkg.collate"}}
    with pytest.raises(ValueError, match=r"dataloader\.collate_fn is not a section"):
        set_dotted(recipe, "dataloader.collate_fn.max_length", 512)
    assert recipe == {"dataloader": {"collate_fn": "pkg.collate"}}


def test_build_recipe_refuses_keys_of_another_template() -> None:
    from vlm_pipeline.train.recipe import RecipeError

    entry = load_catalog().get("qwen3.5-0.8b")
    assert entry is not None
    template = load_template(template_path(project_root(), entry))
    params = LoraParams(advanced={"optimizer.adam_w_mode": True})  # TE FusedAdam only (Parse templates)
    with pytest.raises(RecipeError, match="adam_w_mode"):
        build_recipe(template, params, model_dir="/m", dataset_dir="/d", run_dir="/r", model_id=entry.id)
