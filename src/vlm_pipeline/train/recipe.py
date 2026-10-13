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
from vlm_pipeline.common.decoding import DECODE_OVERRIDES_KEY, decode_overrides_of
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


META_DATASET = "nemo_automodel.components.datasets.vlm.datasets.make_meta_dataset"


def _section(recipe: dict[str, Any], key: str) -> dict[str, Any]:
    value = recipe.get(key)
    if not isinstance(value, dict):
        raise RecipeError(f"template has no {key!r} section")
    return value


def _apply_data(
    r: dict[str, Any], *, dataset_dir: str, train_limit: int | None, train_split: str | None,
    val_dataset_dir: str | None, val_split: str | None, drop_validation: bool, train_meta: str | None,
    val_meta: str | None,
) -> None:  # fmt: skip
    """Point the recipe's dataset sections at the wired data (see ``build_recipe``)."""
    for key in ("dataset", "validation_dataset"):
        if isinstance(r.get(key), dict):
            r[key]["path_or_dataset"] = dataset_dir

    if train_split and not train_meta:
        _section(r, "dataset")["split"] = train_split
    if train_meta:
        r["dataset"] = {"_target_": META_DATASET, "path_or_dataset": train_meta}
        if val_meta and isinstance(r.get("validation_dataset"), dict):
            r["validation_dataset"] = {"_target_": META_DATASET, "path_or_dataset": val_meta}
        elif not val_meta:
            drop_validation = True
    if drop_validation:
        r.pop("validation_dataset", None)
        r.pop("validation_dataloader", None)
    sched = _section(r, "step_scheduler")
    if "validation_dataset" in r and "ckpt_every_steps" in sched:
        sched["val_every_steps"] = sched["ckpt_every_steps"]  # nodes count epochs, not steps: validate per checkpoint
    if not train_meta and val_dataset_dir is not None and isinstance(r.get("validation_dataset"), dict):
        r["validation_dataset"]["path_or_dataset"] = val_dataset_dir
        if val_split:
            r["validation_dataset"]["split"] = val_split
    if train_limit and not train_meta:  # local data: the worker already cut the records
        _section(r, "dataset")["limit_dataset_samples"] = train_limit  # "quick run" sizes of the data node


def build_recipe(
    template: Mapping[str, Any],
    params: LoraParams | FullFinetuneParams,
    *,
    model_dir: str,
    dataset_dir: str,
    run_dir: str,
    class_token_start_idx: int | None = None,
    train_limit: int | None = None,
    train_split: str | None = None,
    val_dataset_dir: str | None = None,
    val_split: str | None = None,
    drop_validation: bool = False,
    train_meta: str | None = None,
    val_meta: str | None = None,
) -> dict[str, Any]:
    """Return a new recipe dict (the template is not modified).

    ``train_split`` / ``val_dataset_dir`` + ``val_split`` come from the Dataset artifacts wired into the node (a
    ``data.split`` node); without them the template's own splits of ``dataset_dir`` are used.
    ``drop_validation``: no val data wired and the dataset has no validation split -> no validation at all (a
    template split that does not exist would crash the job).
    ``train_meta`` / ``val_meta``: local data (container paths of ``make_meta_dataset`` meta files written by the
    train worker from the data node's records); they replace the template's dataset sections, whatever their
    loader (research: templates stay neutral about the data source).
    """
    import copy

    r: dict[str, Any] = copy.deepcopy(dict(template))
    r.pop(DECODE_OVERRIDES_KEY, None)  # ours, not an AutoModel key: applied by the Eval / "Try it" nodes
    _section(r, "model")["pretrained_model_name_or_path"] = model_dir
    _section(r, "processor")["pretrained_model_name_or_path"] = model_dir
    _apply_data(
        r, dataset_dir=dataset_dir, train_limit=train_limit, train_split=train_split, val_dataset_dir=val_dataset_dir,
        val_split=val_split, drop_validation=drop_validation, train_meta=train_meta, val_meta=val_meta,
    )  # fmt: skip
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
    try:
        decode_overrides_of(data)
    except ValueError as exc:  # pydantic.ValidationError: unknown key or out-of-range value
        raise RecipeError(f"template {path.name}: invalid {DECODE_OVERRIDES_KEY}: {exc}") from exc
    return data
