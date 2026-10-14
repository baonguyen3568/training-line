"""Parameters of the training engine, shared by the fine-tune nodes and the train worker (single source).

Host-safe: pydantic only. Defaults and ranges: .team/knowledge/node-params-v1.md §3 (AutoModel v0.6.0 recipes,
S-3 smoke). Per-model defaults (loss, pixels, freezing) come from the model catalog, not from here.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from vlm_pipeline.common.params import Params, ui

DEFAULT_SEED = 1234

LrPreset = Literal["slow", "medium", "fast"]
Epochs = Literal[1, 2, 3, 5]
TrainBlocks = Literal["lm", "projector", "projector_lm", "all"]

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
    # Engine options by dotted key (P016 §3), offered from configs/engine/*.json: checked in train/advanced.py.
    advanced: dict[str, Any] = Field(default_factory=dict, json_schema_extra=ui(advanced=True, widget="advanced"))

    @field_validator("advanced")
    @classmethod
    def _offered_engine_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        from vlm_pipeline.train.advanced import check_advanced

        return check_advanced(value)

    @model_validator(mode="after")
    def _batch_divides(self) -> Self:
        if self.global_batch_size % self.local_batch_size:
            raise ValueError("global_batch_size must be a multiple of local_batch_size")
        return self


class LoraParams(_FinetuneParams):
    lora_dim: int = Field(8, ge=4, le=64, json_schema_extra=ui(advanced=True))
    lora_alpha: int = Field(32, ge=8, le=128, json_schema_extra=ui(advanced=True))
    # QLoRA = this option, not another node (research 2026-10-13: AutoModel create_bnb_config nf4 + double quant,
    # bnb 0.49.2 in the image). Runs on vlm-automodel:26.08-7 (patch 0006: tied lm_head stays bf16; research A/B
    # 2026-10-14: val_loss = LoRA's, -3.4 GB host peak on 2B) for models whose template lists qlora_skip_modules (the
    # vision part stays bf16); the support check refuses it for every other model (CFG-005 not_for_model).
    quantize_base: Literal["none", "nf4"] = Field(
        "none", json_schema_extra=ui(advanced=True, option_keys="quantize_base")
    )

    @property
    def learning_rate(self) -> float:
        return LORA_LR[self.lr_preset]


class FullFinetuneParams(_FinetuneParams):
    # "3 passes" is measured for full SFT only (Parse + CORD, research 001a §10.10); LoRA has no recommendation yet.
    epochs: Epochs = Field(
        3,
        json_schema_extra=ui(order=0, widget="segmented", summary=True, option_keys="epoch_count", recommended=3),
    )
    # ADR-022 #1/#4/#5 (P023 §1): which blocks are updated; the others are frozen. projector / projector_lm need patch
    # 0007 (freeze_config prefixes, vlm-automodel:26.08-8): offered disabled until that image is the default.
    train_blocks: TrainBlocks = Field(
        "lm",
        json_schema_extra=ui(
            advanced=True,
            option_keys="train_blocks",
            disabled_options={
                "projector": "param_unsupported.worker_v1",
                "projector_lm": "param_unsupported.worker_v1",
            },
        ),
    )
    # #5: the vision tower learns at node LR x this (param_group_overrides lr_mult; the recipe adds the lr_scheduler
    # that applies it, research 2026-10-14).
    vision_lr_scale: float = Field(
        0.1, ge=0.05, le=0.2, json_schema_extra=ui(advanced=True, visible_when={"train_blocks": ["all"]})
    )

    @model_validator(mode="before")
    @classmethod
    def _old_freeze_flag(cls, data: Any) -> Any:
        """Workflows saved before train_blocks: freeze_vision_tower true -> lm, false -> all (drop after 1.1)."""
        if isinstance(data, dict) and "freeze_vision_tower" in data:
            data = dict(data)
            frozen = data.pop("freeze_vision_tower")
            data.setdefault("train_blocks", "lm" if frozen else "all")
        return data

    @property
    def learning_rate(self) -> float:
        return FULL_LR[self.lr_preset]


# The rewards NeMo-RL's VLMEnvironment ships (P020 v3: it loads no outside module, so no app code ever scores):
# exact_match = format + exact_alnum, numeric = format + math_expr (sympy equivalence), format = format only.
GrpoReward = Literal["exact_match", "numeric", "format"]


class GrpoParams(Params):
    """train.grpo_vlm (P020): a LoRA adapter trained with GRPO and one of the engine's BUILT-IN rewards (security §B2).

    The reward compares each generated answer with the record's ``target``. Total reward = format_weight x shape
    bonus + correctness_weight x score; defaults follow the engine's geometry3k example (0.1 / 0.9), confirmed by
    research when the NeMo-RL probe runs.
    """

    reward: GrpoReward = Field("exact_match", json_schema_extra=ui(order=0, summary=True, option_keys="grpo_reward"))
    epochs: Epochs = Field(1, json_schema_extra=ui(order=1, widget="segmented", option_keys="epoch_count"))
    lr_preset: LrPreset = Field(
        "medium", json_schema_extra=ui(order=2, widget="segmented", preset=True, option_keys="lr_preset")
    )
    num_generations: int = Field(4, ge=2, le=8, json_schema_extra=ui(advanced=True))  # answers per prompt
    max_new_tokens: int = Field(256, ge=32, le=1024, json_schema_extra=ui(advanced=True))
    kl_coef: float = Field(0.01, ge=0.0, le=1.0, json_schema_extra=ui(advanced=True))
    lora_dim: int = Field(8, ge=4, le=64, json_schema_extra=ui(advanced=True))
    format_weight: float = Field(0.1, ge=0.0, le=1.0, json_schema_extra=ui(advanced=True))
    correctness_weight: float = Field(0.9, ge=0.0, le=1.0, json_schema_extra=ui(advanced=True))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))

    @property
    def learning_rate(self) -> float:
        return LORA_LR[self.lr_preset]


# Preference training moves the policy with far smaller steps than SFT (TRL / Zephyr); provisional LoRA rates until
# research's first VLM DPO run measures them (backend review 2026-10-14).
DPO_LR: dict[LrPreset, float] = {"slow": 5e-6, "medium": 1e-5, "fast": 2e-5}


class DpoParams(Params):
    """train.dpo_vlm (ADR-022 #7, P023 §2): a LoRA adapter trained on preference pairs (records' chosen / rejected).

    Engine: NeMo-RL has DPO for text only (research 2026-10-14); research patches the multimodal path. Until then the
    support check refuses the node (CFG-011 reason engine_no_vlm_dpo). loss = DPO + sft_weight x SFT on the chosen
    answer (0 = plain DPO); the reference model is the base with the adapter off (no second copy in memory).
    """

    beta: float = Field(0.1, ge=0.01, le=1.0, json_schema_extra=ui(order=0))  # TRL default; NeMo-RL's KL penalty
    epochs: Epochs = Field(1, json_schema_extra=ui(order=1, widget="segmented", option_keys="epoch_count"))
    lr_preset: LrPreset = Field(
        "medium", json_schema_extra=ui(order=2, widget="segmented", preset=True, option_keys="lr_preset")
    )
    sft_weight: float = Field(0.0, ge=0.0, le=1.0, json_schema_extra=ui(advanced=True))
    lora_dim: int = Field(8, ge=4, le=64, json_schema_extra=ui(advanced=True))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))

    @property
    def learning_rate(self) -> float:
        return DPO_LR[self.lr_preset]
