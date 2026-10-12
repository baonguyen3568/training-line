"""Build the NeMo AutoModel recipe YAML for one fine-tune node (single translation point params -> engine).

Per-model template: ``configs/train/templates/<catalog id>.yaml`` (derived from the upstream recipe + our smoke runs).
This module only overrides what the node controls: model/data paths, optimiser, batch, epochs, seed, checkpointing,
LoRA vs full fine-tune, freezing. Host-safe (pydantic + pyyaml). Paths given here are CONTAINER paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams

TEMPLATE_DIR = Path("configs") / "train" / "templates"
_LORA_EXCLUDE = ["*vision_tower*", "*vision*", "*visual*", "*image_encoder*", "*lm_head*", "*audio*"]


class RecipeError(ValueError):
    """No template for the model, or the template lacks a section we must override."""


def template_path(project_root: Path, entry: CatalogEntry) -> Path:
    path = project_root / TEMPLATE_DIR / f"{entry.id}.yaml"
    if not path.is_file():
        raise RecipeError(f"no training template for model {entry.id!r}")
    return path


def _section(recipe: dict[str, Any], key: str) -> dict[str, Any]:
    value = recipe.get(key)
    if not isinstance(value, dict):
        raise RecipeError(f"template has no {key!r} section")
    return value


def build_recipe(
    template: Mapping[str, Any],
    params: LoraParams | FullFinetuneParams,
    *,
    model_dir: str,
    dataset_dir: str,
    run_dir: str,
    class_token_start_idx: int | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    """Return a new recipe dict (the template is not modified)."""
    import copy

    r: dict[str, Any] = copy.deepcopy(dict(template))
    _section(r, "model")["pretrained_model_name_or_path"] = model_dir
    _section(r, "processor")["pretrained_model_name_or_path"] = model_dir
    for key in ("dataset", "validation_dataset"):
        if isinstance(r.get(key), dict):
            r[key]["path_or_dataset"] = dataset_dir

    if train_limit:
        _section(r, "dataset")["limit_dataset_samples"] = train_limit  # "quick run" sizes of the data node
    sched = _section(r, "step_scheduler")
    sched["global_batch_size"] = params.global_batch_size
    sched["local_batch_size"] = params.local_batch_size
    sched["num_epochs"] = params.epochs
    sched.pop("max_steps", None)  # the node trains whole epochs; smoke step caps belong to smoke configs

    opt = _section(r, "optimizer")
    opt["lr"] = params.learning_rate
    opt["weight_decay"] = params.weight_decay
    _section(r, "rng")["seed"] = params.seed

    ckpt = _section(r, "checkpoint")
    ckpt.update(
        enabled=True,
        checkpoint_dir=f"{run_dir}/checkpoints",
        model_save_format="safetensors",
        save_consolidated="final",
        max_recent_checkpoints=params.max_recent_checkpoints,
    )
    _section(r, "distributed")["activation_checkpointing"] = params.activation_checkpointing

    if isinstance(params, LoraParams):
        existing = r.get("peft")
        peft: dict[str, Any] = existing if isinstance(existing, dict) else {}
        r["peft"] = {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "match_all_linear": False,
            "exclude_modules": peft.get("exclude_modules", _LORA_EXCLUDE),
            "use_triton": peft.get("use_triton", True),
            "dim": params.lora_dim,
            "alpha": params.lora_alpha,
        }
    else:
        r.pop("peft", None)
        freeze = r.setdefault("freeze_config", {})
        freeze["freeze_vision_tower"] = params.freeze_vision_tower
        freeze.setdefault("freeze_language_model", False)

    loss = r.get("loss_fn")
    if isinstance(loss, dict) and "class_token_start_idx" in loss:
        if class_token_start_idx is None:
            raise RecipeError("Nemotron-Parse loss needs class_token_start_idx from the model config")
        loss["class_token_start_idx"] = class_token_start_idx
    return r


def render_recipe(recipe: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(dict(recipe), sort_keys=False)


def load_template(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RecipeError(f"template {path.name} is not a mapping")
    return data
