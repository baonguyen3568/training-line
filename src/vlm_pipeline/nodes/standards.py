"""Standards as data (P4-2): ``configs/standards/*.yaml`` rules checked against a workflow.

Each rule names the nodes (and param or input port) it applies to, ONE check with ONE threshold, a level and an i18n
message key (``standards.*``). Rules report the single error code ``STD-001`` with the rule id and message key in the
params (fixcode), so a new standard needs YAML + i18n only.

Two phases, fixed by the check kind: ``validate`` runs here, on validated params, wires and the catalogs, before
anything starts; ``data_profile`` checks need the data itself and are applied by the data node's worker to its
``profile.json`` (B-3). A kind is accepted only once its evaluation exists: data never runs ahead of code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_pipeline.common.catalog import CatalogEntry, DatasetEntry, load_catalog, load_dataset_catalog
from vlm_pipeline.nodes.data import DATA_SPLIT
from vlm_pipeline.nodes.rules import origin_model_id
from vlm_pipeline.nodes.source import DATA_HF_LOAD, DATA_LOCAL_FOLDER
from vlm_pipeline.nodes.spec import GraphRule, GraphView, Issue, NodeSpec

STANDARDS_DIR = Path("configs") / "standards"
STANDARD_CODE = "STD-001"
MESSAGE_PATTERN = r"^standards(\.[a-z0-9_]+)+$"  # i18n namespace of standard sentences (also enforced by the app)
Phase = Literal["validate", "data_profile"]
Level = Literal["error", "warn", "info"]
_DATA_SOURCES = frozenset({DATA_HF_LOAD.type_id, DATA_LOCAL_FOLDER.type_id})
_REPO_PATH = re.compile(r"(?:\.team|docs|configs|benchmarks|src|scripts|tests)/[\w./-]*[\w/]")


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Result = tuple[bool, object] | None  # (violated, observed value); None = cannot be evaluated on this graph


class _Context:
    """What a validate-phase check may look at: the graph, the node, the rule's port and the node's params."""

    def __init__(self, view: GraphView, node_id: str, port: str | None, params: BaseModel) -> None:
        self.view, self.node_id, self.port, self.params = view, node_id, port, params

    def upstream_data(self, port: str) -> tuple[str, Any] | None:
        """The data source feeding ``port``; a rule on a data source node itself reads that node."""
        type_id = self.view.specs[self.node_id].type_id
        if type_id in _DATA_SOURCES:
            return type_id, self.params
        return _dataset_upstream(self.view, self.node_id, port)

    def model(self) -> CatalogEntry | None:
        return _model_entry(self.view, self.node_id)


class _Check(_Strict):
    phase: ClassVar[Phase] = "validate"

    def violation(self, ctx: _Context, value: Any) -> Result:
        """Evaluate on the graph; data_profile kinds return None here (the data worker applies them)."""
        return None


# --------------------------------------------------------------------------------------- validate-phase checks
# Every check is a REQUIREMENT: the rule fires when it is not met.


class Range(_Check):
    """The param stays within [min, max]."""

    kind: Literal["range"]
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("range needs min and/or max")
        return self

    def violation(self, ctx: _Context, value: Any) -> Result:
        v = float(value)
        return (self.min is not None and v < self.min) or (self.max is not None and v > self.max), value


class RatioToParam(_Check):
    """param / other stays within [min, max] (e.g. LoRA alpha vs rank)."""

    kind: Literal["ratio_to_param"]
    other: str
    min: float
    max: float

    def violation(self, ctx: _Context, value: Any) -> Result:
        other = float(getattr(ctx.params, self.other))
        ratio = float(value) / other if other else float("inf")
        return not self.min <= ratio <= self.max, round(ratio, 3)


class Equals(_Check):
    """The param equals ``value``."""

    kind: Literal["equals"]
    value: bool | int | float | str

    def violation(self, ctx: _Context, value: Any) -> Result:
        return value != self.value, value


class CatalogFieldIn(_Check):
    """The picked model's catalog field is one of ``allowed``."""

    kind: Literal["catalog_field_in"]
    catalog_field: str
    allowed: tuple[str, ...] = Field(min_length=1)

    def violation(self, ctx: _Context, value: Any) -> Result:
        entry = ctx.model()
        return None if entry is None else (getattr(entry, self.catalog_field) not in self.allowed, entry.id)


class CatalogFieldNotIn(_Check):
    """The picked model's catalog field is none of ``values``."""

    kind: Literal["catalog_field_not_in"]
    catalog_field: str
    values: tuple[str, ...] = Field(min_length=1)

    def violation(self, ctx: _Context, value: Any) -> Result:
        entry = ctx.model()
        return None if entry is None else (getattr(entry, self.catalog_field) in self.values, entry.id)


class EpochsVsSamples(_Check):
    """At most ``max_epochs`` epochs when the training data has fewer than ``small_data_below`` samples."""

    kind: Literal["epochs_vs_samples"]
    max_epochs: int = Field(ge=1)
    small_data_below: int = Field(ge=1)

    def violation(self, ctx: _Context, value: Any) -> Result:
        samples = _sample_count(ctx.upstream_data("train_data"))
        if samples is None:
            return None
        return int(value) > self.max_epochs and samples < self.small_data_below, value


class WiredOrDatasetHasSplit(_Check):
    """The port is wired, or the dataset upstream of the training data has ``split``."""

    kind: Literal["wired_or_dataset_has_split"]
    split: str

    def violation(self, ctx: _Context, value: Any) -> Result:
        if ctx.port is not None and ctx.view.source(ctx.node_id, ctx.port) is not None:
            return False, None
        entry = _catalog_dataset(ctx.upstream_data("train_data"))
        return None if entry is None else (self.split not in entry.splits, entry.id)


class DatasetSplitNotHeldout(_Check):
    """The dataset feeding the port may be trained / calibrated on (not eval-only)."""

    kind: Literal["dataset_split_not_heldout"]

    def violation(self, ctx: _Context, value: Any) -> Result:
        entry = _catalog_dataset(ctx.upstream_data(ctx.port or ""))
        return None if entry is None else (not entry.trainable_splits, entry.id)


class SampleCount(_Check):
    """At least ``below`` samples (known here for catalog datasets; local folders are checked on the profile)."""

    kind: Literal["sample_count"]
    below: int = Field(ge=1)

    def violation(self, ctx: _Context, value: Any) -> Result:
        samples = _sample_count(ctx.upstream_data(ctx.port or "train_data"))
        return None if samples is None else (samples < self.below, samples)


# ------------------------------------------------------------------------------------ data_profile-phase checks


class _ProfileCheck(_Check):
    phase: ClassVar[Phase] = "data_profile"


class ImageStats(_ProfileCheck):
    kind: Literal["image_stats"]
    metric: Literal["broken_pct", "max_aspect_ratio"]
    above: float = Field(ge=0)
    min_short_side_px: int | None = Field(None, ge=1)


class TokenLenStats(_ProfileCheck):
    kind: Literal["token_len_stats"]
    metric: Literal["over_limit_pct"]
    above: float = Field(ge=0)
    limit_from_catalog: Literal["min_answer_tokens"]


class DupRate(_ProfileCheck):
    kind: Literal["dup_rate"]
    method: Literal["sha256_image_text"]
    above: float = Field(ge=0)


class EvalLeak(_ProfileCheck):
    kind: Literal["eval_leak"]
    match: Literal["phash_and_normalized_question", "phash_image_only"]
    above_count: int = Field(ge=0)


Check = Annotated[
    Range
    | RatioToParam
    | Equals
    | CatalogFieldIn
    | CatalogFieldNotIn
    | EpochsVsSamples
    | WiredOrDatasetHasSplit
    | DatasetSplitNotHeldout
    | SampleCount
    | ImageStats
    | TokenLenStats
    | DupRate
    | EvalLeak,
    Field(discriminator="kind"),
]


class AppliesTo(_Strict):
    node: tuple[str, ...] = Field(min_length=1)
    param: str | None = None
    port: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _node_as_tuple(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and isinstance(data.get("node"), str):
            return {**data, "node": (data["node"],)}
        return data

    @model_validator(mode="after")
    def _param_or_port(self) -> Self:
        if self.param is not None and self.port is not None:
            raise ValueError("applies_to takes a param or a port, not both")
        return self


class Rule(_Strict):
    id: str = Field(pattern=r"^[A-Z]+(-[A-Z0-9]+)+$")
    applies_to: AppliesTo
    check: Check
    level: Level
    message: str = Field(pattern=MESSAGE_PATTERN)
    source: str = Field(min_length=1)

    @property
    def phase(self) -> Phase:
        return self.check.phase

    def placeholders(self) -> frozenset[str]:
        """Names a ``standards.*`` text may use: the check's fields plus what every issue carries."""
        return frozenset(self.check.model_dump(exclude={"kind"})) | {"rule", "node", "param", "value"}


class StandardsFile(_Strict):
    schema_version: Literal[1]
    rules: tuple[Rule, ...]


class StandardsError(ValueError):
    """A standards file that does not match the schema, the node registry or the repo (fails loading)."""


def load_standards(root: Path, specs: Iterable[NodeSpec]) -> tuple[Rule, ...]:
    """All rules of ``configs/standards/*.yaml``, checked against the node specs and the repo; raises StandardsError."""
    by_type = {s.type_id: s for s in specs}
    rules: list[Rule] = []
    for path in sorted((root / STANDARDS_DIR).glob("*.yaml")):
        try:
            loaded = StandardsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except ValueError as exc:  # pydantic.ValidationError: unknown kind, typo, two thresholds...
            raise StandardsError(f"{path.name}: {exc}") from exc
        rules += loaded.rules
        for rule in loaded.rules:
            _check_against_repo(rule, by_type, root, path.name)
    duplicates = sorted({r.id for r in rules if [x.id for x in rules].count(r.id) > 1})
    if duplicates:
        raise StandardsError(f"duplicate rule ids: {duplicates}")
    return tuple(rules)


def _check_against_repo(rule: Rule, by_type: Mapping[str, NodeSpec], root: Path, file: str) -> None:
    where = f"{file} {rule.id}"
    for type_id in rule.applies_to.node:
        spec = by_type.get(type_id)
        if spec is None:
            raise StandardsError(f"{where}: unknown node {type_id!r}")
        params = spec.params_schema().get("properties", {})
        names = [n for n in (rule.applies_to.param, getattr(rule.check, "other", None)) if n is not None]
        if missing := [n for n in names if n not in params]:
            raise StandardsError(f"{where}: {type_id} has no param {missing}")
        if rule.applies_to.port is not None and rule.applies_to.port not in spec.inputs:
            raise StandardsError(f"{where}: {type_id} has no input port {rule.applies_to.port!r}")
    for ref in _REPO_PATH.findall(rule.source):
        if not (root / ref).exists():
            raise StandardsError(f"{where}: source path {ref!r} does not exist")


# ------------------------------------------------------------------------------------------- validate phase


def _dataset_upstream(view: GraphView, node_id: str, port: str) -> tuple[str, Any] | None:
    """(source node type, its validated params) of the data source feeding ``port``, through data.split."""
    current = view.source(node_id, port)
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        spec = view.specs[current]
        if spec.type_id == DATA_SPLIT.type_id:
            current = view.source(current, "dataset")
            continue
        return spec.type_id, view.params.get(current)
    return None


def _catalog_dataset(upstream: tuple[str, Any] | None) -> DatasetEntry | None:
    if upstream is None or upstream[0] != DATA_HF_LOAD.type_id or upstream[1] is None:
        return None
    repo = getattr(upstream[1], "repo", None)
    return next((d for d in load_dataset_catalog().datasets if d.repo == repo), None)


def _sample_count(upstream: tuple[str, Any] | None) -> int | None:
    entry = _catalog_dataset(upstream)
    if entry is None:
        return None  # local folder / unknown: the data profile knows
    limit = int(getattr(upstream[1] if upstream else None, "max_samples", 0) or 0)
    return min(entry.num_train, limit) if limit else entry.num_train


def _model_entry(view: GraphView, node_id: str) -> CatalogEntry | None:
    spec = view.specs[node_id]
    port = next((n for n, p in spec.inputs.items() if p.type in {"Model", "Checkpoint", "QuantizedModel"}), None)
    model_id = origin_model_id(view, node_id, port) if port else None
    return load_catalog().get(model_id) if model_id else None


def _violation(rule: Rule, view: GraphView, node_id: str) -> Result:
    """(violated, observed value), or None when the rule cannot be evaluated on this graph (never guessed)."""
    params = view.params.get(node_id)
    if params is None:
        return None  # invalid params: already CFG-005
    value = getattr(params, rule.applies_to.param) if rule.applies_to.param else None
    return rule.check.violation(_Context(view, node_id, rule.applies_to.port, params), value)


def standards_rule(rules: Iterable[Rule]) -> GraphRule:
    """The GraphRule that runs every validate-phase standard (one STD-001 issue per violated rule and node)."""
    active = tuple(r for r in rules if r.phase == "validate")

    def run(view: GraphView) -> Iterator[Issue]:
        for rule in active:
            for node_id, spec in view.specs.items():
                if spec.type_id not in rule.applies_to.node:
                    continue
                result = _violation(rule, view, node_id)
                if result is None or not result[0]:
                    continue
                params = {
                    "rule": rule.id,
                    "message": rule.message,
                    "node": node_id,
                    "param": rule.applies_to.param,  # error catalog: ref param -> shown as the param's label
                    "value": result[1],
                    **rule.check.model_dump(exclude={"kind"}),
                }
                yield Issue(
                    STANDARD_CODE, severity=rule.level, node_id=node_id, port=rule.applies_to.port,
                    param=rule.applies_to.param, params=params,
                )  # fmt: skip

    return run


__all__ = [
    "MESSAGE_PATTERN",
    "STANDARDS_DIR",
    "STANDARD_CODE",
    "Rule",
    "StandardsError",
    "load_standards",
    "standards_rule",
]
