"""Build the NeMo AutoModel recipe YAML for one fine-tune node (single translation point params -> engine).

Per-model template: ``configs/train/templates/<catalog id>.yaml`` (derived from the upstream recipe + our smoke runs).
This module only overrides what the node controls: model/data paths, optimiser, batch, epochs, seed, checkpointing,
LoRA vs full fine-tune, freezing. Host-safe (pydantic + pyyaml). Paths given here are CONTAINER paths.
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_pipeline.common.catalog import CatalogEntry
from vlm_pipeline.common.decoding import DECODE_OVERRIDES_KEY, decode_overrides_of
from vlm_pipeline.common.paths import configs_root
from vlm_pipeline.train.advanced import keys_not_for, set_dotted
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams

TEMPLATE_DIR = Path("configs") / "train" / "templates"
_LORA_EXCLUDE = ["*vision_tower*", "*vision*", "*visual*", "*image_encoder*", "*lm_head*", "*audio*"]


class RecipeError(ValueError):
    """No template for the model, or the template lacks a section we must override."""


def templates_digest(project_root: Path) -> str:
    """12 hex chars over every template (``<name>:<sha256>`` lines, sorted): part of container nodes' cache identity.

    Any template edit (e.g. processor pixel limits) changes what a train / trial / eval run produces with the same node
    params, so it must invalidate the cache. Coarse on purpose: one edit recomputes every container node.
    """
    lines = [
        f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in sorted((configs_root(project_root) / TEMPLATE_DIR).glob("*.yaml"))
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def template_path(project_root: Path, entry: CatalogEntry) -> Path:
    if entry.tier == "user":  # ADR-021: never a bare path: app.user_assets.training_template checks its sha256
        raise RecipeError(f"model {entry.id!r}: a user model's template is read through its verified loader")
    path = configs_root(project_root) / TEMPLATE_DIR / f"{entry.id}.yaml"
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
    if not train_meta and val_dataset_dir is not None and isinstance(r.get("validation_dataset"), dict):
        r["validation_dataset"]["path_or_dataset"] = val_dataset_dir
        if val_split:
            r["validation_dataset"]["split"] = val_split
    if train_limit and not train_meta:  # local data: the worker already cut the records
        _section(r, "dataset")["limit_dataset_samples"] = train_limit  # "quick run" sizes of the data node


# QLoRA base weights (LoraParams.quantize_base="nf4"): AutoModel 26.08 create_bnb_config, as in its
# llama_3_3_70b_instruct_squad_peft_qlora_spark.yaml (bitsandbytes 0.49.2 aarch64 in vlm-automodel:26.08-6).
QLORA_NF4: dict[str, Any] = {
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_quant_storage": "bfloat16",
}
# Ours, not an AutoModel key: the template lists the modules QLoRA must keep in bf16 (vision tower + projector). Their
# names differ per family (Qwen3.5: model.visual; Gemma: model.vision_tower + model.multi_modal_projector), so they
# live with the model's template; lm_head (tied) is skipped by patch 0006 itself (fixcode).
QLORA_SKIP_KEY = "qlora_skip_modules"


def _valid_skip_list(value: object) -> list[str] | None:
    """The one rule for ``qlora_skip_modules``: a non-empty list of non-empty module names, else None."""
    if isinstance(value, list) and value and all(isinstance(m, str) and m for m in value):
        return list(value)
    return None


def qlora_skip_modules(template: Mapping[str, Any]) -> list[str] | None:
    """The template's bf16 modules for QLoRA, or None when the model is not set up for it (no or a malformed list)."""
    return _valid_skip_list(template.get(QLORA_SKIP_KEY))


@functools.cache  # read once per process (the templates ship with the app), like advanced.engine_options
def qlora_models(root: Path | None = None) -> frozenset[str]:
    """Catalog ids whose training template lists ``qlora_skip_modules``: the only models QLoRA is offered for."""
    import yaml

    found = set()
    for path in sorted((configs_root(root) / TEMPLATE_DIR).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and qlora_skip_modules(data) is not None:
            found.add(path.stem)
    return frozenset(found)


# AutoModel applies optimizer.param_group_overrides lr_mult only through its lr_scheduler (research 2026-10-14);
# the templates have none, so the recipe adds one that keeps today's constant LR. Keys: research's CPU run.
LR_SCHEDULER_FOR_LR_MULT: dict[str, Any] = {"lr_decay_style": "constant", "lr_warmup_steps": 0}


@dataclass(frozen=True)
class Blocks:
    """Module prefixes of a model's blocks (families.yaml / the catalog): what train_blocks freezes or trains."""

    vision: tuple[str, ...] = ()
    projector: tuple[str, ...] = ()


def block_freeze(train_blocks: str, blocks: Blocks) -> dict[str, Any]:
    """freeze_config for ``train_blocks`` (patch 0007 keys; ``train_prefixes`` wins over every freeze).

    lm: the vision tower frozen (Qwen's projector sits inside ``visual``; a projector outside it keeps training, as
    before train_blocks; projector / projector_lm need patch 0007 keys, image 26.08-8);
    projector: only the projector (stage 1 of a two-stage recipe; the untied lm_head is frozen too);
    projector_lm: projector + LM; all: nothing frozen.
    """
    if train_blocks not in ("lm", "projector", "projector_lm", "all"):
        raise RecipeError(f"unknown train_blocks {train_blocks!r}")
    if train_blocks in ("projector", "projector_lm") and not blocks.projector:
        raise RecipeError("train_blocks needs the model's projector prefixes (families.yaml / catalog)")
    if train_blocks == "lm":  # exactly the pre-train_blocks freeze: no patch-0007 key, so it runs on every image
        return {"freeze_vision_tower": True, "freeze_language_model": False}
    if train_blocks == "projector":
        return {"freeze_vision_tower": True, "freeze_language_model": True, "freeze_prefixes": ["lm_head"],
                "train_prefixes": list(blocks.projector)}  # fmt: skip
    if train_blocks == "projector_lm":
        return {"freeze_vision_tower": True, "freeze_language_model": False, "train_prefixes": list(blocks.projector)}
    return {"freeze_vision_tower": False, "freeze_language_model": False}


def vision_pattern(blocks: Blocks) -> str:
    """Regex (re.search on parameter names, AutoModel param_group_overrides): the vision tower minus the projector."""
    if not blocks.vision:
        raise RecipeError("vision_lr_scale needs the model's vision prefixes (families.yaml / catalog)")
    vision = "|".join(re.escape(v) for v in blocks.vision)
    inner = [p[len(v) + 1 :] for p in blocks.projector for v in blocks.vision if p.startswith(v + ".")]
    exclude = f"(?!(?:{'|'.join(re.escape(i) for i in inner)})\\.)" if inner else ""
    return f"(?:^|\\.)(?:{vision})\\.{exclude}"


NO_STEP_CADENCE = 1_000_000_000  # ckpt_every_steps that never fires (step % N == N - 1): epoch ends only


def _schedule(r: dict[str, Any], params: LoraParams | FullFinetuneParams, trial_steps: int | None) -> None:
    """The step scheduler: whole epochs, saved + validated at epoch ends; a quick trial: a few steps, no validation."""
    sched = _section(r, "step_scheduler")
    sched["global_batch_size"] = params.global_batch_size
    sched["local_batch_size"] = params.local_batch_size
    sched["num_epochs"] = params.epochs
    sched.pop("max_steps", None)  # the node trains whole epochs; smoke step caps belong to smoke configs
    # Save + validate at epoch boundaries only (research 2026-10-13): save_checkpoint_every_epoch saves at each epoch
    # end (the last step included) and the engine validates exactly when it saves. The step cadence is switched off
    # explicitly: leaving ckpt_every_steps out is NOT "per epoch" in AutoModel 26.08, whose StepSchedulerConfig
    # defaults it to 100 (R2 saved + validated at steps 99 and 199 on top of the epoch ends, research 2026-10-13).
    sched["ckpt_every_steps"] = NO_STEP_CADENCE
    sched.pop("val_every_steps", None)
    sched["save_checkpoint_every_epoch"] = True
    if trial_steps:  # "Chạy thử nhanh" (MODEL-005, a run option): a few steps to see it runs; no validation
        sched["max_steps"] = trial_steps
        sched["num_epochs"] = 1
        sched["ckpt_every_steps"] = trial_steps  # saved at the last step even though the epoch never ends
        r.pop("validation_dataset", None)
        r.pop("validation_dataloader", None)


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
    model_id: str | None = None,
    blocks: Blocks = Blocks(),  # noqa: B008 - frozen dataclass, an immutable default
    trial_steps: int | None = None,
) -> dict[str, Any]:
    """Return a new recipe dict (the template is not modified).

    ``train_split`` / ``val_dataset_dir`` + ``val_split`` come from the Dataset artifacts wired into the node (a
    ``data.split`` node); without them the template's own splits of ``dataset_dir`` are used.
    ``drop_validation``: no val data wired and the dataset has no validation split -> no validation at all (a
    template split that does not exist would crash the job).
    ``train_meta`` / ``val_meta``: local data (container paths of ``make_meta_dataset`` meta files written by the
    train worker from the data node's records); they replace the template's dataset sections, whatever their
    loader (research: templates stay neutral about the data source).
    ``model_id``: the catalog id of the template, to refuse Advanced keys it does not have (P016 §3).
    """
    import copy

    r: dict[str, Any] = copy.deepcopy(dict(template))
    r.pop(DECODE_OVERRIDES_KEY, None)  # ours, not an AutoModel key: applied by the Eval / "Try it" nodes
    skip = r.pop(QLORA_SKIP_KEY, None)
    _section(r, "model")["pretrained_model_name_or_path"] = model_dir
    _section(r, "processor")["pretrained_model_name_or_path"] = model_dir
    _apply_data(
        r, dataset_dir=dataset_dir, train_limit=train_limit, train_split=train_split, val_dataset_dir=val_dataset_dir,
        val_split=val_split, drop_validation=drop_validation, train_meta=train_meta, val_meta=val_meta,
    )  # fmt: skip
    if model_id is not None and (wrong := keys_not_for(model_id, params.advanced)):  # also outside the canvas (CLI)
        raise RecipeError(f"advanced options not in the {model_id} template: {wrong}")
    for key, value in sorted(params.advanced.items()):  # P016 §3: after the template, before the node's own fields
        try:
            set_dotted(r, key, value)
        except ValueError as exc:
            raise RecipeError(str(exc)) from exc
    _schedule(r, params, trial_steps)

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

    r.pop("quantization", None)  # only the node turns 4-bit on (QLoRA below), never a template; never for full FT
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
        if params.quantize_base == "nf4":  # QLoRA: AutoModel's BitsAndBytes section (its DGX Spark QLoRA example)
            if (modules := _valid_skip_list(skip)) is None:
                raise RecipeError(f"QLoRA is not set up for this model: its template has no {QLORA_SKIP_KEY}")
            r["quantization"] = {**QLORA_NF4, "llm_int8_skip_modules": modules}  # the vision part stays bf16
    else:
        r.pop("peft", None)
        r["freeze_config"] = {**r.get("freeze_config", {}), **block_freeze(params.train_blocks, blocks)}
        if params.train_blocks == "all" and params.vision_lr_scale != 1:
            _section(r, "optimizer")["param_group_overrides"] = [
                {"pattern": vision_pattern(blocks), "lr_mult": params.vision_lr_scale}
            ]
            r.setdefault("lr_scheduler", dict(LR_SCHEDULER_FOR_LR_MULT))  # lr_mult is applied by the scheduler only

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
    return template_from_bytes(path.read_bytes(), path.name)


def template_from_bytes(raw: bytes, name: str) -> dict[str, Any]:
    """A template parsed from its bytes (the verified loader of user templates hashes these same bytes)."""
    import yaml

    path = Path(name)
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RecipeError(f"template {path.name} is not a mapping")
    try:
        decode_overrides_of(data)
    except ValueError as exc:  # pydantic.ValidationError: unknown key or out-of-range value
        raise RecipeError(f"template {path.name}: invalid {DECODE_OVERRIDES_KEY}: {exc}") from exc
    return data
