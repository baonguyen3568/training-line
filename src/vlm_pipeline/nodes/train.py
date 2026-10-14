"""Fine-tune nodes (category ``train``). Heavy: one container job each (P009 §1)."""

from __future__ import annotations

from vlm_pipeline.nodes.spec import NodeSpec, PortSpec
from vlm_pipeline.train.config import DpoParams, FullFinetuneParams, GrpoParams, LoraParams

_FINETUNE_INPUTS = {
    "base_model": PortSpec(type="Model"),
    "train_data": PortSpec(type="Dataset"),
    "val_data": PortSpec(type="Dataset", optional=True),
}

# TODO(M1, bao-backend): set `image` to the vlm-automodel digest from docker/IMAGES.lock when the worker is wired.
TRAIN_LORA_VLM = NodeSpec(
    type_id="train.lora_vlm",
    version="1.0.0",
    category="train",
    icon="train",
    inputs=_FINETUNE_INPUTS,
    outputs={"adapter": PortSpec(type="Adapter")},
    params_model=LoraParams,
    runner="container",
)

TRAIN_SFT_VLM = NodeSpec(
    type_id="train.sft_vlm",
    version="1.0.0",
    category="train",
    icon="train",
    inputs=_FINETUNE_INPUTS,
    outputs={"model": PortSpec(type="Checkpoint")},
    params_model=FullFinetuneParams,
    runner="container",
)

# P020: "coming soon" in 1.0 (no handler; the support check refuses it with its reason, engine_single_gpu).
TRAIN_GRPO_VLM = NodeSpec(
    type_id="train.grpo_vlm",
    version="1.0.0",
    category="train",
    icon="train",
    inputs=_FINETUNE_INPUTS,
    outputs={"adapter": PortSpec(type="Adapter")},  # a LoRA adapter: Merge / Eval / Export reuse it (P020 §6)
    params_model=GrpoParams,
    runner="container",
)

# ADR-022 #7: refused until research's multimodal DPO patch lands (support check, CFG-011 reason engine_no_vlm_dpo).
TRAIN_DPO_VLM = NodeSpec(
    type_id="train.dpo_vlm",
    version="1.0.0",
    category="train",
    icon="train",
    inputs=_FINETUNE_INPUTS,
    outputs={"adapter": PortSpec(type="Adapter")},
    params_model=DpoParams,
    runner="container",
)
