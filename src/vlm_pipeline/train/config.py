"""Parameters of the training engine, shared by the fine-tune nodes and the train worker (single source).

Host-safe: pydantic only. Defaults and ranges: .team/knowledge/node-params-v1.md §3 (AutoModel v0.6.0 recipes,
S-3 smoke). Per-model defaults (loss, pixels, freezing) come from the model catalog, not from here.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from vlm_pipeline.common.params import Params, ui

DEFAULT_SEED = 1234

LrPreset = Literal["slow", "medium", "fast"]
Epochs = Literal[1, 3, 5]

# Learning rate per preset: full fine-tune follows the Nemotron-Parse / Qwen3-VL recipes (1e-5),
# LoRA follows the qwen3_8 LoRA recipe (1e-4).
FULL_LR: dict[LrPreset, float] = {"slow": 5e-6, "medium": 1e-5, "fast": 2e-5}
LORA_LR: dict[LrPreset, float] = {"slow": 5e-5, "medium": 1e-4, "fast": 2e-4}


class _FinetuneParams(Params):
    epochs: Epochs = Field(
        3, json_schema_extra=ui(order=0, widget="segmented", summary=True, option_keys="epoch_count")
    )
    lr_preset: LrPreset = Field(
        "medium", json_schema_extra=ui(order=1, widget="segmented", preset=True, summary=True, option_keys="lr_preset")
    )
    global_batch_size: int = Field(8, ge=4, le=64, json_schema_extra=ui(advanced=True))
    local_batch_size: int = Field(1, ge=1, le=4, json_schema_extra=ui(advanced=True))
    weight_decay: float = Field(0.01, ge=0.0, le=0.1, json_schema_extra=ui(advanced=True))
    activation_checkpointing: bool = Field(True, json_schema_extra=ui(advanced=True))
    max_recent_checkpoints: int = Field(2, ge=1, le=5, json_schema_extra=ui(advanced=True))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))

    @model_validator(mode="after")
    def _batch_divides(self) -> Self:
        if self.global_batch_size % self.local_batch_size:
            raise ValueError("global_batch_size must be a multiple of local_batch_size")
        return self


class LoraParams(_FinetuneParams):
    lora_dim: int = Field(8, ge=4, le=64, json_schema_extra=ui(advanced=True))
    lora_alpha: int = Field(32, ge=8, le=128, json_schema_extra=ui(advanced=True))

    @property
    def learning_rate(self) -> float:
        return LORA_LR[self.lr_preset]


class FullFinetuneParams(_FinetuneParams):
    freeze_vision_tower: bool = Field(True, json_schema_extra=ui(advanced=True))

    @property
    def learning_rate(self) -> float:
        return FULL_LR[self.lr_preset]
