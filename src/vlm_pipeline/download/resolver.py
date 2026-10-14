"""A user-added Hugging Face model -> a catalog entry + a training template, from data only (ADR-021, P021 H1).

Nothing here imports, loads or runs anything from the repo: the plan job (egress proxy) wrote the repo's small JSON
files and its file list into its run dir; this module reads them with size/depth caps, refuses any sign of remote code
(security checklist 2026-10-14 §4), maps ``architectures[0]`` to a family of ``configs/catalog/families.yaml``
(research) and builds the same ``CatalogEntry`` the vetted catalog uses, so every check downstream is shared
(ADR-021 §6). The family's recipe is the training template, with the family's module lists.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_pipeline.common.catalog import REPO_ID_PATTERN, REVISION_PATTERN, CatalogEntry
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.paths import configs_root
from vlm_pipeline.common.runinfo import read_small_json

FAMILIES_FILE = Path("configs") / "catalog" / "families.yaml"
JSON_MAX = 1 << 20  # each config / processor / tokenizer json (security §5)
INDEX_MAX = 8 << 20  # model.safetensors.index.json
MAX_DEPTH = 32
MAX_KEY = 256
MAX_NAME = 80
MAX_TOTAL_BYTES = 2 * 10**12  # a safetensors total_size beyond this is not a real model (bounded int, security §5)
# The JSON files the plan job fetches and that are scanned for remote code (security §4).
CONFIG_FILES = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)
_CODE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+--[A-Za-z0-9_.]+$")  # "org/repo--module.Class"
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_ID_PART = re.compile(r"[^a-z0-9.]+")
_PYTHON_SUFFIXES = (".py", ".pyc", ".pyo", ".pyw")  # any file Python can load, any case (security H1)
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


class ModelArchUnsupportedError(VlmPipelineError):
    """``architectures[0]`` is not a family the app knows how to train natively (MODEL-003)."""

    code: ClassVar[str] = "MODEL-003"


class ModelNeedsRemoteCodeError(VlmPipelineError):
    """The repo needs its own Python code: blocked outside the catalog (ADR-021 §3, MODEL-004)."""

    code: ClassVar[str] = "MODEL-004"


class RepoFilesInvalidError(ValueError):
    """A file the plan job wrote is missing, too large, not plain JSON, or too deep (never trusted, never guessed)."""


# --------------------------------------------------------------------------------------------- families (research)
class RamFormula(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    a: float = Field(gt=0)
    b: float = Field(ge=0)
    valid_up_to_params_b: float = Field(gt=0)


class Family(BaseModel):
    """One row of configs/catalog/families.yaml (field meanings: its header)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    model_types: tuple[str, ...]
    architectures: tuple[str, ...] = Field(min_length=1)
    auto_class: Literal["AutoModelForImageTextToText"]  # native only: AutoModel means remote code
    attn: Literal["eager", "sdpa"]
    verified: Literal["A", "B", "C"]
    train_node: Literal["train.lora_vlm"]
    recipe: str = Field(pattern=r"^configs/train/templates/[a-z0-9][a-z0-9.\-]*\.yaml$")
    freeze_vision_tower: bool
    lora_exclude: tuple[str, ...] = Field(min_length=1)
    qlora_skip_modules: tuple[str, ...] = ()
    loss: Literal["masked_ce"]
    image_policy: Literal["qwen_pixels"]  # processor_default joins when its CatalogEntry policy exists
    min_pixels: int = Field(gt=0)
    max_pixels: int = Field(gt=0)
    ram_gb_train: RamFormula


class Families(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    families: tuple[Family, ...]

    @model_validator(mode="after")
    def _one_family_per_architecture(self) -> Self:
        names = [a for f in self.families for a in f.architectures]
        if len(names) != len(set(names)):
            raise ValueError("an architecture is listed by two families")
        return self

    def for_architecture(self, architecture: str) -> Family | None:
        return next((f for f in self.families if architecture in f.architectures), None)


@functools.cache  # the file ships with the app: read once per process
def load_families(root: Path | None = None) -> Families:
    path = configs_root(root) / FAMILIES_FILE
    if not path.is_file():
        return Families(schema_version=1, families=())
    return Families.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------------- reading the files
def _reject_constant(name: str) -> object:
    raise ValueError(f"non-JSON constant {name}")


def _strict_loads(data: bytes) -> object:
    value = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    _check_shape(value, depth=0)
    return value


def _check_shape(value: object, *, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise ValueError(f"nesting deeper than {MAX_DEPTH}")
    if isinstance(value, dict):
        for key, child in value.items():
            if len(key) > MAX_KEY:
                raise ValueError("object key too long")
            _check_shape(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_shape(child, depth=depth + 1)


def read_repo_json(path: Path, max_bytes: int = JSON_MAX) -> object:
    """A JSON file the plan job wrote: no symlink, regular file, size cap, plain JSON, bounded depth / key length."""
    try:
        return read_small_json(path, max_bytes=max_bytes, loads=_strict_loads)
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise RepoFilesInvalidError(f"{path.name}: {exc}") from None


def clean_name(value: object) -> str:
    """An architecture / model type / license string as it may reach a log or the UI (≤ 80, [A-Za-z0-9_.-])."""
    text = _UNSAFE_NAME.sub("", value)[:MAX_NAME] if isinstance(value, str) else ""
    return text or "unknown"


# --------------------------------------------------------------------------------------------- remote code
def _walk_signs(value: object, where: str, signs: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "auto_map":
                signs.append(f"auto_map:{where}")
            elif key == "trust_remote_code":
                signs.append(f"trust_remote_code:{where}")
            _walk_signs(child, where, signs)
    elif isinstance(value, list):
        for child in value:
            _walk_signs(child, where, signs)
    elif isinstance(value, str) and _CODE_REF.match(value):
        signs.append(f"code_ref:{where}")


def remote_code_signs(configs: Mapping[str, object], files: Sequence[str]) -> list[str]:
    """Every sign that the repo needs its own code (security §4); empty = none of the static signs.

    ``auto_map`` anywhere (any depth, any of the JSON files), a ``trust_remote_code`` key, an ``org/repo--module.Class``
    value, or a ``*.py`` file in the repo. The fifth sign (a class not native in the train image) is the family table
    here and the image lookup in the job.
    """
    signs: list[str] = []
    for name in sorted(configs):
        _walk_signs(configs[name], name, signs)
    if any(f.lower().endswith(_PYTHON_SUFFIXES) for f in files):
        signs.append("py_file")
    return sorted(set(signs))


# --------------------------------------------------------------------------------------------- facts + entry
@dataclass(frozen=True)
class ModelFacts:
    architecture: str  # architectures[0], cleaned: what errors show
    architectures: tuple[str, ...]  # all of them, cleaned (each must belong to the family)
    model_type: str
    torch_dtype: str


def model_facts(config: object) -> ModelFacts:
    """The few config.json fields the resolver reads (types checked, strings cleaned); nothing else is looked at."""
    if not isinstance(config, dict):
        raise RepoFilesInvalidError("config.json is not an object")
    archs = config.get("architectures")
    names = tuple(clean_name(a) for a in archs[:8]) if isinstance(archs, list) and archs else ("unknown",)
    dtype = config.get("torch_dtype", config.get("dtype"))
    return ModelFacts(
        architecture=names[0],
        architectures=names,
        model_type=clean_name(config.get("model_type")),
        torch_dtype=dtype if isinstance(dtype, str) and dtype in _DTYPE_BYTES else "bfloat16",
    )


def params_b_from_index(index: object, torch_dtype: str) -> float | None:
    """Billions of parameters from ``model.safetensors.index.json`` ``metadata.total_size`` (bounded), else None."""
    meta = index.get("metadata") if isinstance(index, dict) else None
    total = meta.get("total_size") if isinstance(meta, dict) else None
    if isinstance(total, bool) or not isinstance(total, int) or not 0 < total <= MAX_TOTAL_BYTES:
        return None
    return round(total / _DTYPE_BYTES.get(torch_dtype, 2) / 1e9, 2)


def user_model_id(repo: str, revision: str) -> str:
    """``user--<org>--<name>--<sha8>``: stable, lowercase, never a catalog id (those have no ``--``)."""
    org, name = repo.split("/", 1)
    parts = [_ID_PART.sub("-", p.lower()).strip("-") or "x" for p in (org, name)]
    return f"user--{parts[0]}--{parts[1]}--{revision[:8]}"


def ram_upper_bound(family: Family, params_b: float) -> float:
    """The family's RAM bound from research (LoRA bf16, bs 1, pixel cap); +20 % past the sizes it was fitted on."""
    ram = family.ram_gb_train
    gb = ram.a * params_b + ram.b
    return round(gb * (1.2 if params_b > ram.valid_up_to_params_b else 1.0), 1)


def resolve_family(
    configs: Mapping[str, object], files: Sequence[str], *, repo: str, revision: str, families: Families
) -> tuple[Family, ModelFacts]:
    """The family of a repo, or the catalog error that says why it cannot be trained (MODEL-004 before MODEL-003)."""
    if "config.json" not in configs:
        raise RepoFilesInvalidError("the repo has no config.json")
    if remote_code_signs(configs, files):
        raise ModelNeedsRemoteCodeError("the repo needs its own code", params={"repo": repo, "revision": revision[:12]})
    facts = model_facts(configs["config.json"])
    family = families.for_architecture(facts.architecture)
    # transformers builds the class from model_type (CONFIG_MAPPING), not from architectures: both must be the
    # family's, and every listed architecture too, or the load would fail late (security H1).
    if family is not None and (
        facts.model_type not in family.model_types or any(a not in family.architectures for a in facts.architectures)
    ):
        family = None
    if family is None:
        raise ModelArchUnsupportedError(
            "architecture without a native family",
            params={"repo": repo, "revision": revision[:12], "architecture": facts.architecture,
                    "model_type": facts.model_type},
        )  # fmt: skip
    return family, facts


def user_entry(
    *, repo: str, revision: str, family: Family, params_b: float, disk_gb: float, license_id: str, gated: bool
) -> CatalogEntry:
    """The ``CatalogEntry`` of a user-added model: unverified, estimated RAM, no remote code, LoRA only."""
    if not re.fullmatch(REPO_ID_PATTERN, repo) or not re.fullmatch(REVISION_PATTERN, revision):
        raise ValueError("repo / revision not in their pinned form")
    return CatalogEntry(
        id=user_model_id(repo, revision), repo=repo, revision=revision, display_name=repo.split("/", 1)[1],
        vendor="user", tier="user", gated=gated, remote_code="none", auto_class=family.auto_class,
        license=clean_name(license_id), commercial="conditional", attribution="", security="unreviewed",
        params_b=params_b, disk_gb=max(0.01, math.ceil(disk_gb * 100) / 100), train_node=family.train_node,
        ram_gb_train=ram_upper_bound(family, params_b), ram_source="estimated", sft_trainable_pct_min=0,
        sft_trainable_pct_max=0, attn=family.attn, loss=family.loss, freeze_vision_tower=family.freeze_vision_tower,
        min_pixels=family.min_pixels, max_pixels=family.max_pixels, image_policy=family.image_policy,
        image_max_tokens=0, video=False, video_fps=0, min_answer_tokens=1024, quant_fp8="unsupported",
        serve_engine="hf", task_prompt_fixed=False, gallery_gates=(), min_image=DEFAULT_TRAIN_IMAGE,
    )  # fmt: skip


def render_template(family: Family, root: Path | None = None) -> dict[str, Any]:
    """The training template of a user model: the family's recipe with the family's module lists and pixel caps.

    Model/processor paths are filled by ``build_recipe`` as for any catalog model.
    """
    template: dict[str, Any] = copy.deepcopy(
        yaml.safe_load((configs_root(root) / family.recipe).read_text(encoding="utf-8"))
    )
    processor = template.setdefault("processor", {})
    processor.update(min_pixels=family.min_pixels, max_pixels=family.max_pixels)
    template.setdefault("peft", {})["exclude_modules"] = list(family.lora_exclude)
    template.setdefault("freeze_config", {})["freeze_vision_tower"] = family.freeze_vision_tower
    if family.qlora_skip_modules:
        template["qlora_skip_modules"] = list(family.qlora_skip_modules)
    else:
        template.pop("qlora_skip_modules", None)
    return template


__all__ = [
    "CONFIG_FILES",
    "Families",
    "Family",
    "ModelArchUnsupportedError",
    "ModelFacts",
    "ModelNeedsRemoteCodeError",
    "RepoFilesInvalidError",
    "clean_name",
    "load_families",
    "model_facts",
    "params_b_from_index",
    "ram_upper_bound",
    "read_repo_json",
    "remote_code_signs",
    "render_template",
    "resolve_family",
    "user_entry",
    "user_model_id",
]
