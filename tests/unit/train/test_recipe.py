from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams
from vlm_pipeline.train.recipe import RecipeError, build_recipe, load_template, render_recipe, template_path

ROOT = project_root()


def _tpl(model_id: str) -> dict:  # type: ignore[type-arg]
    entry = load_catalog().get(model_id)
    assert entry is not None
    return load_template(template_path(ROOT, entry))


def test_full_finetune_parse_overrides_only_node_fields() -> None:
    params = FullFinetuneParams(epochs=1, lr_preset="slow", global_batch_size=8, seed=7, max_recent_checkpoints=3)
    r = build_recipe(_tpl("nemotron-parse-v1.1"), params, model_dir="/w/m", dataset_dir="/w/d", run_dir="/w/r",
                     class_token_start_idx=52315)  # fmt: skip
    assert r["model"]["pretrained_model_name_or_path"] == "/w/m"
    assert r["processor"]["pretrained_model_name_or_path"] == "/w/m"
    assert r["dataset"]["path_or_dataset"] == r["validation_dataset"]["path_or_dataset"] == "/w/d"
    assert r["step_scheduler"]["num_epochs"] == 1
    assert "max_steps" not in r["step_scheduler"]
    assert r["optimizer"]["lr"] == 5e-6
    assert r["rng"]["seed"] == 7
    assert r["checkpoint"]["checkpoint_dir"] == "/w/r/checkpoints"
    assert r["checkpoint"]["max_recent_checkpoints"] == 3
    assert r["checkpoint"]["save_consolidated"] == "final"
    assert "peft" not in r
    assert r["freeze_config"]["freeze_vision_tower"] is True
    assert r["loss_fn"]["class_token_start_idx"] == 52315
    assert r["recipe"] == "FinetuneRecipeForVLM"  # untouched template content
    assert "pretrained_model_name_or_path: /w/m" in render_recipe(r)


def test_parse_requires_class_token_threshold() -> None:
    with pytest.raises(RecipeError, match="class_token_start_idx"):
        build_recipe(_tpl("nemotron-parse-v1.1"), FullFinetuneParams(), model_dir="m", dataset_dir="d", run_dir="r")


def test_lora_sets_peft_from_params() -> None:
    r = build_recipe(_tpl("cosmos-reason2-2b"), LoraParams(lora_dim=16, lora_alpha=64), model_dir="m",
                     dataset_dir="d", run_dir="r")  # fmt: skip
    assert r["peft"]["dim"] == 16
    assert r["peft"]["alpha"] == 64
    assert "*vision*" in r["peft"]["exclude_modules"]
    assert r["optimizer"]["lr"] == 1e-4


def test_missing_template(tmp_path: Path) -> None:
    entry = load_catalog().get("qwen3.5-9b")
    assert entry is not None
    with pytest.raises(RecipeError, match="no training template"):
        template_path(tmp_path, entry)


def test_train_limit_sets_dataset_limit_only() -> None:
    r = build_recipe(_tpl("cosmos-reason2-2b"), LoraParams(), model_dir="m", dataset_dir="d", run_dir="r",
                     train_limit=100)  # fmt: skip
    assert r["dataset"]["limit_dataset_samples"] == 100


def test_decode_overrides_never_reach_automodel() -> None:
    template = {**_tpl("nemotron-parse-v1.1"), "decode_overrides": {"repetition_penalty": 1.0}}
    r = build_recipe(template, FullFinetuneParams(), model_dir="/w/m", dataset_dir="/w/d", run_dir="/w/r",
                     class_token_start_idx=50000)  # fmt: skip
    assert "decode_overrides" not in r
    assert template["decode_overrides"] == {"repetition_penalty": 1.0}  # input untouched
