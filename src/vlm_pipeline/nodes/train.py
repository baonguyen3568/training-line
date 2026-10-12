"""Fine-tune nodes (category ``train``). Heavy: one container job each (P009 §1)."""

from __future__ import annotations

from vlm_pipeline.nodes.spec import NodeSpec, PortSpec
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams

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
