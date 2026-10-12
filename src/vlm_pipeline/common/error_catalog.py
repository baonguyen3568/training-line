"""Error catalog loader: ``common/error_catalog.yaml`` -> validated, frozen models (Proposal 007 R9).

The catalog holds codes, severities, params and actions only; user text lives in ``i18n/{vi,en}.yaml``.
Validation is strict (``extra="forbid"``) so a typo like ``prevent_in_prefight`` fails at load time, and
cross-field rules (declared params, actions for errors, variants) are checked by model validators.
Host-side only (pydantic + pyyaml); container workers use ``common.errors`` which stays stdlib-only.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from importlib import resources
from typing import Annotated, Any, Literal, Protocol, Self

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from vlm_pipeline.common.errors import VlmPipelineError

CODE_PATTERN = r"^[A-Z]+-\d{3}$"
FALLBACK_CODE = "SYS-000"


class CatalogError(VlmPipelineError):
    """The error catalog file itself is malformed (developer error, caught by unit tests)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Param(_Frozen):
    name: str
    type: Literal["any", "enum", "bool"] = "any"
    values: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _enum_has_values(self) -> Self:
        if self.type == "enum" and not self.values:
            raise ValueError(f"enum param {self.name!r} needs values")
        return self


def _param_from_name(raw: object) -> object:
    # YAML allows a bare name for plain params and a mapping for typed (enum/bool) ones.
    return {"name": raw} if isinstance(raw, str) else raw


When = Mapping[str, bool | str | list[str]]


class Action(_Frozen):
    id: str
    kind: Literal["patch", "resume", "navigate"]
    confirm: bool
    target: str | None = None
    patch: tuple[Mapping[str, Any], ...] | None = None
    when: When = Field(default_factory=dict)


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


def available_actions(error: _ErrorLike) -> list[str]:
    """Catalog action ids offered for ``error`` (filtered by each action's ``when``).

    ``resumable`` comes from the error object (set by the executor from a real checkpoint) and defaults to False,
    so ``resume_checkpoint`` is never offered without one. Unknown codes get the ``SYS-000`` actions.
    """
    catalog = load_catalog()
    entry = catalog.get(error.code) or catalog[FALLBACK_CODE]
    env: dict[str, object] = {**error.params, "resumable": bool(getattr(error, "resumable", False))}
    return [a.id for a in entry.actions if _matches(a.when, env)]


__all__ = ["Action", "CatalogError", "Entry", "Param", "Variant", "available_actions", "load_catalog"]
