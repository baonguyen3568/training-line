"""Advanced engine options of the train nodes (P016 §3): offered keys only, typed, bounded, per model template."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.advanced import ERROR_TYPE, check_advanced, keys_not_for, set_dotted
from vlm_pipeline.train.config import LoraParams
from vlm_pipeline.train.recipe import build_recipe, load_template, template_path


def test_offered_values_pass() -> None:
    values = {"optimizer.eps": 1e-8, "model.backend.attn": "flex", "dataloader.num_workers": 2, "rng.ranked": True}
    assert check_advanced(values) == values
    assert LoraParams(advanced=values).advanced == values


@pytest.mark.parametrize(
    ("values", "rule"),
    [
        ({"model._target_": "os.system"}, "not_offered"),  # denied keys never reach the file
        ({"optimizer.lr": 1.0}, "not_offered"),  # the node's own field
        ({"dataloader.num_workers": 512}, "range"),  # shared-host DoS
        ({"dataloader.num_workers": True}, "type"),  # bool is not an int here
        ({"rng.ranked": 1}, "type"),
        ({"model.backend.attn": "flash3"}, "choice"),
        ({"model.backend.attn": "${oc.env:HF_TOKEN}"}, "resolver"),
        ({f"k{i}": 1 for i in range(101)}, "too_many"),
    ],
)
def test_bad_values_are_refused(values: dict[str, Any], rule: str) -> None:
    with pytest.raises(PydanticCustomError) as err:
        check_advanced(values)
    assert [p["rule"] for p in err.value.context["problems"]] == [rule]  # type: ignore[index]
    with pytest.raises(ValidationError) as verr:
        LoraParams(advanced=values)
    assert verr.value.errors()[0]["type"] == ERROR_TYPE


def test_every_bad_key_is_listed_and_unknown_keys_are_echoed_only_when_key_shaped() -> None:
    bad = {"dataloader.num_workers": 512, "model.backend.attn": "flash3", "nope.key": 1, "<img src=x>": 1, "a.b\n": 1}
    with pytest.raises(PydanticCustomError) as err:
        check_advanced(bad)
    problems = err.value.context["problems"]  # type: ignore[index]
    assert problems == [
        {"key": "dataloader.num_workers", "rule": "range"},
        {"key": "model.backend.attn", "rule": "choice"},
        {"key": "nope.key", "rule": "not_offered"},
        {"key": "", "rule": "not_offered"},  # never echo an arbitrary string from the file
        {"key": "", "rule": "not_offered"},  # a trailing newline is not a key ($ alone would let it through)
    ]


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
    assert (info.value.code, info.value.params["key"], info.value.params["rule"]) == (
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


@pytest.mark.parametrize(("model_id", "ok"), [("qwen3.5-2b", True), ("qwen3.5-9b", True), ("gemma-4-e4b-it", False)])
def test_qlora_runs_only_for_models_whose_template_keeps_the_vision_part_bf16(model_id: str, ok: bool) -> None:
    """research A/B 2026-10-14 (26.08-7 PASS): nf4 is offered for every model, refused where the template has no
    qlora_skip_modules (a quantized vision tower fails at step 0)."""
    from vlm_pipeline.app.workers import QloraNotForModelError, _check_train
    from vlm_pipeline.graph.worker import SupportQuery
    from vlm_pipeline.nodes import REGISTRY
    from vlm_pipeline.train.recipe import qlora_models

    spec, model = REGISTRY.get("train.lora_vlm"), load_catalog().get(model_id)
    assert spec is not None
    assert model is not None
    assert (model_id in qlora_models()) is ok
    query = SupportQuery("t", spec, LoraParams(quantize_base="nf4"), model, {})
    if ok:
        _check_train(query)
        return
    with pytest.raises(QloraNotForModelError) as info:
        _check_train(query)
    assert info.value.code == "CFG-005"
    assert info.value.params == {"node": "t", "param": "quantize_base", "value": "nf4", "rule": "not_for_model",
                                 "key": ""}  # fmt: skip


def test_nf4_is_no_longer_a_disabled_option() -> None:
    from vlm_pipeline.train.config import LoraParams as Params

    extra = Params.model_json_schema()["properties"]["quantize_base"]
    assert "nf4" not in str(extra.get("disabled_options", {}))


def test_qlora_needs_the_image_with_patch_0006() -> None:
    """security 2026-10-14: nf4 fails at step 0 before vlm-automodel:26.08-7 (tied lm_head); a rollback of the default
    image must also take QLoRA back out (disabled_options + the support check), or this test fails."""
    from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE

    name, tag = DEFAULT_TRAIN_IMAGE.split(":")
    assert name == "vlm-automodel"
    assert tag.startswith("26.08-")
    assert int(tag.rsplit("-", 1)[1]) >= 7
