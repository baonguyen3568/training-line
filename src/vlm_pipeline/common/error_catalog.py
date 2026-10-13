"""Error catalog loader: ``common/error_catalog.yaml`` -> validated, frozen models (Proposal 007 R9).

The catalog holds codes, severities, params and actions only; user text lives in ``i18n/{vi,en}.yaml``.
Validation is strict (``extra="forbid"``) so a typo like ``prevent_in_prefight`` fails at load time, and
cross-field rules (declared params, actions for errors, variants) are checked by model validators.
Host-side only (pydantic + pyyaml); container workers use ``common.errors`` which stays stdlib-only.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Annotated, Any, Literal, Protocol, Self

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from vlm_pipeline.common.errors import VlmPipelineError

_log = logging.getLogger(__name__)

CODE_PATTERN = r"^[A-Z]+-\d{3}$"
FALLBACK_CODE = "SYS-000"


class CatalogError(VlmPipelineError):
    """The error catalog file itself is malformed (developer error, caught by unit tests)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


ParamRef = Literal["node", "node_list", "port", "param", "port_type"]


class Param(_Frozen):
    name: str
    type: Literal["any", "enum", "bool"] = "any"
    values: tuple[str, ...] = ()
    # The raw value is an internal id (node id, port name, param name, port type id): ``display_params`` swaps it
    # for its localized label before rendering, the raw id stays on the Issue (node_id/port/param) for the UI.
    ref: ParamRef | None = None

    @model_validator(mode="after")
    def _enum_has_values(self) -> Self:
        if self.type == "enum" and not self.values:
            raise ValueError(f"enum param {self.name!r} needs values")
        return self


def _param_from_name(raw: object) -> object:
    # YAML allows a bare name for plain params and a mapping for typed (enum/bool) ones.
    return {"name": raw} if isinstance(raw, str) else raw


When = Mapping[str, bool | str | list[str]]


class PatchOp(_Frozen):
    """One node-param change. ``scale`` multiplies the current value (ints are floored)."""

    param: str
    op: Literal["set", "scale"]
    value: bool | int | float | str


class Action(_Frozen):
    id: str
    kind: Literal["patch", "resume", "navigate"]
    confirm: bool
    target: str | None = None
    patch: tuple[PatchOp, ...] | None = None
    when: When = Field(default_factory=dict)
    # Actions that lower peak memory; when none applies to the node, the error record gets already_optimized=True.
    effect: Literal["memory"] | None = None


class Variant(_Frozen):
    when: When
    key: str


class Entry(_Frozen):
    code: str = Field(pattern=CODE_PATTERN)
    slug: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    severity: Literal["error", "warn", "info"]
    prevent_in_preflight: bool
    params: tuple[Annotated[Param, BeforeValidator(_param_from_name)], ...] = ()
    actions: tuple[Action, ...] = ()
    match: tuple[Mapping[str, str], ...] = ()
    variant_by: str | None = None
    variants: tuple[Variant, ...] = ()
    deprecated: bool = False

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Self:
        names = {p.name for p in self.params}
        if self.severity == "error" and not self.actions:
            raise ValueError("severity=error needs at least one action")
        for a in self.actions:
            if unknown := set(a.when) - names:
                raise ValueError(f"action {a.id!r} `when` uses undeclared params {sorted(unknown)}")
        if self.variant_by and self.variants:
            raise ValueError("use either variant_by or variants, not both")
        if self.variant_by and self.variant_by not in names:
            raise ValueError(f"variant_by {self.variant_by!r} is not a declared param")
        for v in self.variants:
            if unknown := set(v.when) - names:
                raise ValueError(f"variant {v.key!r} uses undeclared params {sorted(unknown)}")
        return self


@functools.cache
def load_catalog() -> dict[str, Entry]:
    """Load and validate ``common/error_catalog.yaml`` (cached)."""
    text = resources.files("vlm_pipeline.common").joinpath("error_catalog.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise CatalogError("error_catalog.yaml must be a mapping of code -> entry")
    catalog: dict[str, Entry] = {}
    for code, body in raw.items():
        try:
            catalog[str(code)] = Entry.model_validate({"code": str(code), **body})
        except (ValidationError, TypeError) as exc:
            raise CatalogError(f"{code}: {exc}") from exc
    if FALLBACK_CODE not in catalog:
        raise CatalogError(f"fallback {FALLBACK_CODE} missing")
    slugs = [e.slug for e in catalog.values()]
    if len(slugs) != len(set(slugs)):
        raise CatalogError("duplicate slug")
    return catalog


class _ErrorLike(Protocol):
    code: str
    params: Mapping[str, object]


def _matches(when: When, env: Mapping[str, object]) -> bool:
    for name, expected in when.items():
        allowed = expected if isinstance(expected, list) else [expected]
        if env.get(name) not in allowed:
            return False
    return True


@dataclass(frozen=True)
class Range:
    """Allowed numeric range of a node param; ``*_exclusive`` mirrors ``Field(gt=..)`` / ``Field(lt=..)``."""

    low: float | None = None
    high: float | None = None
    low_exclusive: bool = False
    high_exclusive: bool = False

    def contains(self, value: float) -> bool:
        if self.low is not None and (value <= self.low if self.low_exclusive else value < self.low):
            return False
        return not (self.high is not None and (value >= self.high if self.high_exclusive else value > self.high))


@dataclass(frozen=True)
class NodeContext:
    """Current params of the failing node plus the constraints of its NodeSpec.

    Preferred: ``NodeContext.from_model(params, NodeSpec.params_model)`` -- a patch is offered only if the patched
    params pass ``params_model.model_validate`` (pydantic is the source of truth: Optional fields, custom validators,
    multipleOf, ...). ``from_schema`` is the fallback when only the JSON schema is available (e.g. API clients).
    """

    params: Mapping[str, object]
    ranges: Mapping[str, Range] = field(default_factory=dict)
    choices: Mapping[str, tuple[object, ...]] = field(default_factory=dict)  # JSON-schema enum / Literal values
    params_model: type[BaseModel] | None = None

    @classmethod
    def from_model(cls, params: Mapping[str, object], params_model: type[BaseModel]) -> NodeContext:
        return cls(params=params, params_model=params_model)

    @classmethod
    def from_schema(cls, params: Mapping[str, object], json_schema: Mapping[str, Any]) -> NodeContext:
        ranges: dict[str, Range] = {}
        choices: dict[str, tuple[object, ...]] = {}
        defs = json_schema.get("$defs") or {}
        for name, raw in (json_schema.get("properties") or {}).items():
            prop = _resolve_schema_prop(raw, defs)
            if prop is None:  # several non-null alternatives: constraints unknown, leave unconstrained
                continue
            if "enum" in prop:
                choices[name] = tuple(prop["enum"])
            elif "const" in prop:
                choices[name] = (prop["const"],)
            low_ex, high_ex = "exclusiveMinimum" in prop, "exclusiveMaximum" in prop
            low = prop.get("exclusiveMinimum", prop.get("minimum"))
            high = prop.get("exclusiveMaximum", prop.get("maximum"))
            if low is not None or high is not None:
                ranges[name] = Range(low, high, low_ex, high_ex)
        return cls(params=params, ranges=ranges, choices=choices)


def _resolve_schema_prop(raw: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Follow ``$ref`` (Enum classes) and unwrap ``Optional[X]`` (``anyOf``/``oneOf`` with one non-null branch)."""
    for key in ("anyOf", "oneOf"):
        if key in raw:
            branches = [b for b in raw[key] if b.get("type") != "null"]
            if len(branches) != 1:
                return None
            raw = {**{k: v for k, v in raw.items() if k != key}, **branches[0]}
    if "$ref" in raw:
        raw = {**defs.get(str(raw["$ref"]).rsplit("/", 1)[-1], {}), **raw}
    return raw


def apply_patch(op: PatchOp, current: object) -> object:
    """New value of one param after ``op``.

    THE ONLY implementation: graph/app must call this when a user applies an action, so "would this button change
    anything" (``available_actions``) always matches what is really applied.

    ``scale`` floors ints and never takes a positive int below 1 (a batch of 0 is never a valid suggestion);
    scaling a non-number is a no-op.
    """
    if op.op == "set":
        return op.value
    if isinstance(current, bool) or not isinstance(current, int | float) or isinstance(op.value, bool | str):
        return current
    scaled = current * op.value
    if isinstance(current, int):
        floored = math.floor(scaled)
        return max(floored, 1) if current >= 1 else floored
    return scaled


def apply_patches(action: Action, params: Mapping[str, object]) -> dict[str, object]:
    """Params after applying every op of ``action`` (unknown params are left untouched)."""
    out = dict(params)
    for op in action.patch or ():
        if op.param in out:
            out[op.param] = apply_patch(op, out[op.param])
    return out


def _op_applies(op: PatchOp, node: NodeContext) -> bool:
    """Offered only if the param exists, the value changes, and the new value is allowed by the NodeSpec."""
    if op.param not in node.params:
        return False
    current = node.params[op.param]
    new = apply_patch(op, current)
    if new == current:
        return False
    allowed = node.choices.get(op.param)
    if allowed is not None and new not in allowed:  # e.g. method enum ["full"] -> no "use LoRA" button
        return False
    rng = node.ranges.get(op.param)
    return not (
        rng is not None and isinstance(new, int | float) and not isinstance(new, bool) and not rng.contains(new)
    )


def _applicable(action: Action, node: NodeContext | None) -> bool:
    if node is None or not action.patch:
        return True
    if not all(_op_applies(op, node) for op in action.patch):
        return False
    if node.params_model is not None:
        try:
            node.params_model.model_validate(apply_patches(action, node.params))
        except ValidationError:
            return False
    return True


def _offered(error: _ErrorLike, node: NodeContext | None) -> tuple[Entry, list[Action]]:
    catalog = load_catalog()
    entry = catalog.get(error.code) or catalog[FALLBACK_CODE]
    env: dict[str, object] = {**error.params, "resumable": bool(getattr(error, "resumable", False))}
    return entry, [a for a in entry.actions if _matches(a.when, env) and _applicable(a, node)]


def available_actions(error: _ErrorLike, node: NodeContext | None = None) -> list[str]:
    """Catalog action ids offered for ``error``.

    Filtered by each action's ``when`` and, when ``node`` is given, by whether its patch would actually change the
    node (param exists, value differs, within NodeSpec bounds) so the UI never offers a button that does nothing.
    ``resumable`` comes from the error object (set by the executor from a real checkpoint) and defaults to False,
    so ``resume_checkpoint`` is never offered without one. Unknown codes get the ``SYS-000`` actions.
    """
    return [a.id for a in _offered(error, node)[1]]


def resolved_actions(
    error: _ErrorLike, node: NodeContext | None = None, node_id: str | None = None
) -> list[dict[str, Any]]:
    """The offered actions, ready for the UI's quick-fix buttons (G-3, bao-frontend): no catalog logic on the client.

    Same filter as ``available_actions``. A param patch carries, per param, ``from`` (the value the server saw) and
    ``value`` (the FINAL value from ``apply_patch``: scaled, floored, inside the NodeSpec bounds). The client applies
    ``value`` only while the param still equals ``from``; otherwise the suggestion is stale (the user edited the node)
    and is hidden until the next validation. A patch action without ops (``remove_step``, ``auto_rename_ids``...) is
    a graph operation identified by its id: ``patch`` is None. Without ``node`` a param patch has no final value, so
    it is left out (the client must never compute one).
    """
    out: list[dict[str, Any]] = []
    for a in _offered(error, node)[1]:
        item: dict[str, Any] = {"id": a.id, "kind": a.kind, "confirm": a.confirm, "target": a.target, "patch": None}
        if node_id is not None:
            item["node_id"] = node_id
        if a.kind == "patch" and a.patch:
            if node is None:
                continue
            item["patch"] = [
                {"param": op.param, "from": node.params[op.param], "value": apply_patch(op, node.params[op.param])}
                for op in a.patch
            ]
        out.append(item)
    return out


def already_optimized(error: _ErrorLike, node: NodeContext | None) -> bool:
    """True when the code has memory-reducing actions but none applies to this node (UI: error_card.already_optimized).

    Computed per error at runtime and sent in the error record; never stored in the catalog. Without a node context
    (``node=None``) nothing can be judged, so it is always False.
    """
    if node is None:
        return False
    entry, offered = _offered(error, node)
    has_memory_actions = any(a.effect == "memory" for a in entry.actions)
    return has_memory_actions and not any(a.effect == "memory" for a in offered)


@dataclass(frozen=True)
class Labels:
    """Resolvers from internal ids to localized labels (wired by the app with i18n + the workflow being shown).

    ``port``/``param`` labels need the node the issue points at (``node_id``); unknown ids fall back to the raw id.
    """

    node: Callable[[str], str | None]  # node id -> node title, e.g. "Fine-tune nhẹ (LoRA)"
    port: Callable[[str, str], str | None]  # (node id, port name) -> port label, e.g. "Model gốc"
    param: Callable[[str, str], str | None]  # (node id, param name) -> param label
    port_type: Callable[[str], str | None]  # port type id -> label, e.g. "Dữ liệu"


def variant_key(code: str, params: Mapping[str, object]) -> str | None:
    """Text variant of ``error.<code>.what`` for these raw params (``variant_by`` / ``variants``), None if plain."""
    entry = load_catalog().get(code)
    if entry is None:
        return None
    if entry.variant_by:
        name = entry.variant_by
        value = params.get(name)
        declared = next(p for p in entry.params if p.name == name)  # checked when the catalog loads
        if declared.type == "bool":  # i18n keys: <name> / not_<name>; "true"/"false" may arrive via JSON from workers
            truthy = value is True or (isinstance(value, str) and value.lower() == "true")
            return name if truthy else f"not_{name}"
        return None if value is None else str(value)
    if not entry.variants:
        return None
    key = next((v.key for v in entry.variants if _matches(v.when, params)), None)
    if key is None:  # a code with variants has no plain `what` text: never show a raw key
        _log.warning(
            "error %s: no text variant matches params %s; using %r", code, sorted(params), entry.variants[0].key
        )
        key = entry.variants[0].key
    return key


def display_params(
    code: str, params: Mapping[str, object], labels: Labels, *, node_id: str | None
) -> dict[str, object]:
    """Params with id-valued entries replaced by labels, for rendering ``error.<code>.*`` (never for the Issue)."""
    entry = load_catalog().get(code)
    refs = {p.name: p.ref for p in entry.params if p.ref} if entry else {}
    out = dict(params)
    for name, ref in refs.items():
        raw = out.get(name)
        if not isinstance(raw, str) or not raw:
            continue
        if ref == "node":
            out[name] = labels.node(raw) or raw
        elif ref == "node_list":
            out[name] = ", ".join(labels.node(n.strip()) or n.strip() for n in raw.split(","))
        elif ref == "port_type":
            out[name] = labels.port_type(raw) or raw
        elif node_id is not None:  # port / param: relative to the node the issue points at
            resolve = labels.port if ref == "port" else labels.param
            out[name] = resolve(node_id, raw) or raw
    return out


__all__ = [
    "Action",
    "CatalogError",
    "Entry",
    "Labels",
    "NodeContext",
    "Param",
    "ParamRef",
    "PatchOp",
    "Range",
    "Variant",
    "already_optimized",
    "apply_patch",
    "apply_patches",
    "available_actions",
    "display_params",
    "load_catalog",
    "resolved_actions",
    "variant_key",
]
