"""Invariants of the per-model training templates (P4-2: standards the user cannot change are tests, not rules).

Full fine-tune on one GPU needs fp32 master weights: AutoModel skips FSDP2's mixed-precision policy on a single GPU, and
plain AdamW on bf16 parameters drops most updates (001a §10.9: P-3 changed 5.3% of decoder elements in 300 steps; P-3b-A
with TE FusedAdam master weights changed 63.8% in 100). The optimizer is not a user knob (P016 bans `_target_` in
Advanced), so every full-fine-tune template must carry it.
"""

from __future__ import annotations

import pytest

from vlm_pipeline.common.catalog import CatalogEntry, load_catalog
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.train.config import FullFinetuneParams
from vlm_pipeline.train.recipe import build_recipe, load_template, template_path

FP32_MASTER_OPTIMIZER = "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
FULL_FINETUNE = [m for m in load_catalog().models if m.train_node == "train.sft_vlm"]


@pytest.mark.parametrize("entry", FULL_FINETUNE, ids=lambda m: m.id)
def test_full_finetune_template_keeps_fp32_master_weights(entry: CatalogEntry) -> None:
    template = load_template(template_path(project_root(), entry))
    recipe = build_recipe(template, FullFinetuneParams(), model_dir="/m", dataset_dir="/d", run_dir="/r",
                          class_token_start_idx=50000)  # fmt: skip
    optimizer = recipe["optimizer"]  # after build_recipe: the node must not drop what the template sets
    assert optimizer["_target_"] == FP32_MASTER_OPTIMIZER, entry.id
    assert optimizer.get("master_weights") is True, entry.id


def test_there_is_a_full_finetune_model_to_check() -> None:
    assert FULL_FINETUNE, "no catalog model uses train.sft_vlm: the invariant above checks nothing"
