"""Model catalog loader: ``configs/catalog/models.yaml`` (owner: bao-research; data, not code).

Strict on purpose: an unknown or misspelt field, an unknown enum value or an unpinned revision fails loading, so a
catalog edit cannot silently change a per-model default. Host-side (pydantic + pyyaml).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from vlm_pipeline.common.paths import configs_root

CATALOG_PATH = Path("configs") / "catalog" / "models.yaml"
DATASETS_PATH = Path("configs") / "catalog" / "datasets.yaml"

# Hub repo id "org/name". No segment may start with "." so "../x" or "org/.." can never reach a path or URL.
REPO_ID_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
# A dataset whose parts are under many licences (e.g. OCRBench): no single text can be vendored, so it may only be
# evaluated on, never trained on or redistributed (bao-security). Export lineage never contains it.
UNVERIFIED_LICENSE = "mixed-unverified"


def _snapshot_relative(path: str) -> str:
    if ".." in path.split("/"):
        raise ValueError(f"{path!r} leaves the snapshot")
    return path


# Relative path inside the dataset snapshot: no leading "/", no ".." segment.
DataFile = Annotated[str, Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$"), AfterValidator(_snapshot_relative)]

# "user": a model the user added (ADR-021); never in configs/catalog/models.yaml (Catalog refuses it there).
Tier = Literal["default", "template", "catalog", "baseline", "rnd", "user"]


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
    vendor: str  # slug, a key of Catalog.vendors
    tier: Tier
    gated: bool
    remote_code: Literal["none", "pinned_prepared"]
    # transformers class that loads the model (export model card, P012); AutoModel = the repo's auto_map (remote code)
    auto_class: Literal["AutoModelForImageTextToText", "AutoModel"]
    license: str
    commercial: Literal["yes", "no", "conditional"]
    attribution: str
    # "unreviewed": a user-added model (ADR-021), never in the catalog file
    security: Literal["approved", "approved_with_conditions", "conditional", "rejected", "unreviewed"]
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
    # processor_default: a user-added model whose own processor sizes images (ADR-021; never in the catalog file)
    image_policy: Literal["fixed", "qwen_pixels", "token_budget", "processor_default"]
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
    # Module blocks for train_blocks / vision_lr_scale (research 2026-10-14): a family of configs/catalog/families.yaml
    # (its vision/projector prefixes), or both prefix lists spelled out (e.g. Nemotron-Parse), never both ways.
    family: str | None = Field(None, pattern=r"^[a-z0-9_]+$")
    vision_prefixes: tuple[str, ...] = ()
    projector_prefixes: tuple[str, ...] = ()

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
    def _blocks_one_way(self) -> Self:
        explicit = bool(self.vision_prefixes) or bool(self.projector_prefixes)
        if self.family is not None and explicit:
            raise ValueError(f"{self.id}: a family or explicit prefixes, not both")
        if explicit and not (self.vision_prefixes and self.projector_prefixes):
            raise ValueError(f"{self.id}: explicit prefixes need both vision_prefixes and projector_prefixes")
        if not all(p and " " not in p for p in (*self.vision_prefixes, *self.projector_prefixes)):
            raise ValueError(f"{self.id}: a prefix is an empty or spaced module path")
        return self

    @model_validator(mode="after")
    def _auto_class_matches_remote_code(self) -> Self:
        # AutoModel only resolves through the repo's auto_map, i.e. pinned remote code; native classes need none.
        if (self.auto_class == "AutoModel") != (self.remote_code == "pinned_prepared"):
            raise ValueError(
                f"{self.id}: auto_class AutoModel goes with remote_code pinned_prepared (and only with it)"
            )
        return self

    @model_validator(mode="after")
    def _image_policy_consistent(self) -> Self:
        # Rules from the catalog header: the policy decides which image widget exists and the train/serve parity key.
        pixels = (self.min_pixels, self.max_pixels)
        if self.image_policy == "fixed":
            ok = self.fixed_image_hw is not None and pixels == (0, 0) and self.image_max_tokens == 0
        elif self.image_policy == "qwen_pixels":
            ok = 0 < self.min_pixels < self.max_pixels and self.fixed_image_hw is None and self.image_max_tokens == 0
        elif self.image_policy == "processor_default":
            ok = pixels == (0, 0) and self.fixed_image_hw is None and self.image_max_tokens == 0
        else:
            ok = self.image_max_tokens > 0 and pixels == (0, 0) and self.fixed_image_hw is None
        if not ok:
            raise ValueError(f"{self.id}: fields inconsistent with image_policy={self.image_policy!r}")
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    budget_note: str = ""
    vendors: dict[str, str]  # slug -> publisher as people write it ("NVIDIA"); a proper noun, not in i18n
    models: tuple[CatalogEntry, ...]

    @model_validator(mode="after")
    def _unique_and_one_default(self) -> Self:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model id in catalog")
        if any(m.tier == "user" or m.security == "unreviewed" for m in self.models):
            raise ValueError("tier user / security unreviewed are for user-added models, never in the catalog file")
        if [m.tier for m in self.models].count("default") != 1:
            raise ValueError("catalog needs exactly one model with tier: default")
        if unknown := sorted({m.vendor for m in self.models} - set(self.vendors)):
            raise ValueError(f"vendor not in catalog vendors: {unknown}")
        return self

    def vendor_name(self, entry: CatalogEntry) -> str:
        """The publisher shown to users ("NVIDIA"), from the one ``vendors`` map."""
        return self.vendors[entry.vendor]

    def get(self, model_id: str) -> CatalogEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def default(self) -> CatalogEntry:
        return next(m for m in self.models if m.tier == "default")

    def selectable(self) -> tuple[CatalogEntry, ...]:
        """Models a user may pick (security did not reject them). Tier-based Simple/Advanced filtering is UI-side."""
        return tuple(m for m in self.models if m.security != "rejected")


# User-added models (ADR-021): records and rendered templates under the data root, never in configs/.
USER_MODELS_REL = Path(".cache") / "user_assets" / "models"
_user_lookup: Callable[[str], CatalogEntry | None] | None = None


def set_user_model_lookup(lookup: Callable[[str], CatalogEntry | None] | None) -> None:
    """The app registers how a ``user--…`` id becomes an entry (its validated record); common never imports app."""
    global _user_lookup  # noqa: PLW0603 - one process-wide source, set once at app startup
    _user_lookup = lookup


def model_entry(model_id: str | None) -> CatalogEntry | None:
    """The one lookup for a model id: the vetted catalog, then a user-added model (ADR-021 §6: one check path)."""
    if not model_id:
        return None
    entry = load_catalog().get(model_id)
    if entry is None and model_id.startswith("user--") and _user_lookup is not None:
        entry = _user_lookup(model_id)
    return entry


def load_catalog_file(path: Path) -> Catalog:
    return Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@functools.cache
def load_catalog() -> Catalog:
    """The project catalog (cached for the process)."""
    return load_catalog_file(configs_root() / CATALOG_PATH)


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
    # Files the download step must verify (snapshot-relative path -> sha256), e.g. a security condition.
    files_sha256: dict[DataFile, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(default_factory=dict)
    # Eval data the app extracts into plain files (download.extracted): sha256 of the MANIFEST.sha256 a correct,
    # deterministic extraction writes, so the installed tree is bound to a known result (security 2026-10-13).
    extracted_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _unverified_licence_is_eval_only(self) -> Self:
        if self.license == UNVERIFIED_LICENSE and (
            self.use != "eval_only" or self.commercial == "yes" or not self.files_sha256
        ):
            raise ValueError(
                f"{self.id}: license {UNVERIFIED_LICENSE} requires use eval_only, commercial != yes and files_sha256"
            )
        return self

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


def license_ids_needing_text(catalog: Catalog, datasets: DatasetCatalog) -> set[str]:
    """Licence ids that must have a vendored, approved text (configs/licenses): every one an export can contain.

    Models, separately licensed model parts and datasets, except the eval-only ``mixed-unverified`` ones.
    """
    ids = {m.license for m in catalog.models} | {p.license for m in catalog.models for p in m.model_parts}
    return ids | {d.license for d in datasets.datasets if d.license != UNVERIFIED_LICENSE}


@functools.cache
def load_dataset_catalog() -> DatasetCatalog:
    """The project dataset catalog (cached for the process)."""
    path = configs_root() / DATASETS_PATH
    return DatasetCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
