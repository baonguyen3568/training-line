"""Model catalog loader: ``configs/catalog/models.yaml`` (owner: bao-research; data, not code).

Strict on purpose: an unknown or misspelt field, an unknown enum value or an unpinned revision fails loading, so a
catalog edit cannot silently change a per-model default. Host-side (pydantic + pyyaml).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from vlm_pipeline.common.paths import project_root

CATALOG_PATH = Path("configs") / "catalog" / "models.yaml"
DATASETS_PATH = Path("configs") / "catalog" / "datasets.yaml"

# Hub repo id "org/name". No segment may start with "." so "../x" or "org/.." can never reach a path or URL.
REPO_ID_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"

Tier = Literal["default", "template", "catalog", "baseline", "rnd"]


# Required notice sentences (Proposal 012 §2.2): research-written, security-approved, copied into an export's NOTICE.
# The export fills {repo} {revision} {date} {tool}; any other {placeholder} is kept visible, never dropped.
Notices = tuple[str, ...]


class ModelPart(BaseModel):
    """Part of a model snapshot licensed separately from the weights (e.g. Parse's CC-BY-4.0 tokenizer)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    license: str = Field(min_length=1)
    notices: Notices = ()


def _empty_as_none(value: object) -> object:
    return None if value == [] else value  # the YAML writes `[]` for "not fixed"


class CatalogEntry(BaseModel):
    """One vetted model. Field meanings: header comment of configs/catalog/models.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]*$")
    repo: str = Field(pattern=REPO_ID_PATTERN)
    revision: str = Field(pattern=REVISION_PATTERN)
    display_name: str = Field(min_length=1)  # proper name only; tagline in i18n model_catalog.<id>.tagline
    vendor: str
    tier: Tier
    gated: bool
    remote_code: Literal["none", "pinned_prepared"]
    license: str
    commercial: Literal["yes", "no", "conditional"]
    attribution: str
    security: Literal["approved", "approved_with_conditions", "conditional", "rejected"]
    params_b: float = Field(gt=0)
    disk_gb: float = Field(gt=0)
    train_node: Literal["train.sft_vlm", "train.lora_vlm", "none"]
    ram_gb_train: float = Field(gt=0)
    ram_source: Literal["measured", "estimated"]
    # Expected trainable share for full SFT (guard VLM_EXPECTED_TRAINABLE_PCT, patch 0002). 0/0 = no expectation yet.
    sft_trainable_pct_min: float = Field(ge=0, le=100)
    sft_trainable_pct_max: float = Field(ge=0, le=100)
    attn: Literal["eager", "sdpa"]
    loss: Literal["masked_ce", "nemotron_parse"]
    freeze_vision_tower: bool
    min_pixels: int = Field(ge=0)
    max_pixels: int = Field(ge=0)
    fixed_image_hw: Annotated[tuple[int, int] | None, BeforeValidator(_empty_as_none)] = None
    image_policy: Literal["fixed", "qwen_pixels", "token_budget"]
    image_max_tokens: int = Field(ge=0)
    video: bool
    video_fps: float = Field(ge=0)
    min_answer_tokens: int = Field(gt=0)
    quant_fp8: Literal["official", "experimental", "unsupported"]
    serve_engine: Literal["vllm", "hf", "unknown"]
    task_prompt_fixed: bool
    gallery_gates: tuple[str, ...]
    min_image: str
    notices: Notices = ()
    model_parts: tuple[ModelPart, ...] = ()

    @property
    def sft_trainable_pct(self) -> tuple[float, float] | None:
        """``(min, max)`` expected trainable share for full SFT, ``None`` when not known yet."""
        if (self.sft_trainable_pct_min, self.sft_trainable_pct_max) == (0, 0):
            return None
        return self.sft_trainable_pct_min, self.sft_trainable_pct_max

    def supports_train_node(self, node_type: str) -> bool:
        """Whether ``node_type`` is the fine-tune path vetted for this model (CFG-010).

        Only that path has a measured RAM budget and a smoke run. Single rule for graph validation and the worker.
        """
        return node_type == self.train_node  # "none" matches no node

    @model_validator(mode="after")
    def _trainable_pct_range(self) -> Self:
        low, high = self.sft_trainable_pct_min, self.sft_trainable_pct_max
        if (low, high) != (0, 0) and not 0 < low < high <= 100:
            raise ValueError(f"{self.id}: sft_trainable_pct must be 0/0 (unknown) or 0 < min < max <= 100")
        return self

    @model_validator(mode="after")
    def _image_policy_consistent(self) -> Self:
        # Rules from the catalog header: the policy decides which image widget exists and the train/serve parity key.
        pixels = (self.min_pixels, self.max_pixels)
        if self.image_policy == "fixed":
            ok = self.fixed_image_hw is not None and pixels == (0, 0) and self.image_max_tokens == 0
        elif self.image_policy == "qwen_pixels":
            ok = 0 < self.min_pixels < self.max_pixels and self.fixed_image_hw is None and self.image_max_tokens == 0
        else:
            ok = self.image_max_tokens > 0 and pixels == (0, 0) and self.fixed_image_hw is None
        if not ok:
            raise ValueError(f"{self.id}: fields inconsistent with image_policy={self.image_policy!r}")
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    budget_note: str = ""
    models: tuple[CatalogEntry, ...]

    @model_validator(mode="after")
    def _unique_and_one_default(self) -> Self:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model id in catalog")
        if [m.tier for m in self.models].count("default") != 1:
            raise ValueError("catalog needs exactly one model with tier: default")
        return self

    def get(self, model_id: str) -> CatalogEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def default(self) -> CatalogEntry:
        return next(m for m in self.models if m.tier == "default")

    def selectable(self) -> tuple[CatalogEntry, ...]:
        """Models a user may pick (security did not reject them). Tier-based Simple/Advanced filtering is UI-side."""
        return tuple(m for m in self.models if m.security != "rejected")


def load_catalog_file(path: Path) -> Catalog:
    return Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@functools.cache
def load_catalog() -> Catalog:
    """The project catalog (cached for the process)."""
    return load_catalog_file(project_root() / CATALOG_PATH)


# ------------------------------------------------------------------------------------------------- datasets


class DatasetEntry(BaseModel):
    """One vetted dataset. Field meanings: header comment of configs/catalog/datasets.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.\-]*$")
    repo: str = Field(pattern=REPO_ID_PATTERN)
    revision: str = Field(pattern=REVISION_PATTERN)
    display_name: str = Field(min_length=1)
    license: str
    commercial: Literal["yes", "no", "conditional"]
    attribution: str = Field(min_length=1)
    security: Literal["approved", "approved_with_conditions", "conditional", "rejected"]
    use: Literal["train", "eval_only"]
    splits: tuple[str, ...] = Field(min_length=1)
    heldout_splits: tuple[str, ...] = ()
    num_train: int = Field(ge=0)
    format: str
    loader: str = Field(pattern=r"^[a-z_][a-z0-9_]*$")
    gated: bool
    disk_gb: float = Field(gt=0)
    notices: Notices = ()

    @model_validator(mode="after")
    def _heldout_are_splits(self) -> Self:
        if unknown := set(self.heldout_splits) - set(self.splits):
            raise ValueError(f"{self.id}: heldout_splits not in splits: {sorted(unknown)}")
        return self

    @property
    def trainable_splits(self) -> tuple[str, ...]:
        """Splits that may be used for training/calibration (never a held-out split, never an eval-only set)."""
        if self.use == "eval_only":
            return ()
        return tuple(s for s in self.splits if s not in self.heldout_splits)


class DatasetCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    datasets: tuple[DatasetEntry, ...]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [d.id for d in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate dataset id in catalog")
        return self

    def get(self, dataset_id: str) -> DatasetEntry | None:
        return next((d for d in self.datasets if d.id == dataset_id), None)


@functools.cache
def load_dataset_catalog() -> DatasetCatalog:
    """The project dataset catalog (cached for the process)."""
    path = project_root() / DATASETS_PATH
    return DatasetCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
