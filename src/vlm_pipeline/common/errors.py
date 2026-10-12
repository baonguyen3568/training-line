"""Shared error types and process exit codes.

Stdlib-only at import time (pyyaml is imported lazily by ``load_catalog``). Exceptions carry only a catalog
``code`` plus ``params``; user-facing text lives in ``i18n/{vi,en}.yaml`` and is
rendered at the app edge (Proposal 007 R9). Codes follow ``<GROUP>-<NNN>`` from the error catalog.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from importlib import resources
from typing import Any, ClassVar, Literal, Protocol


class ExitCode(IntEnum):
    """Process exit codes shared by preflight, job runner, worker and dashboard (ADR-005 D8, Proposal 007)."""

    OK = 0
    NO_MEM = 2
    NO_DISK = 3
    LOCK_BUSY = 4
    MEM_PREEMPT = 75  # worker stopped by the memory watchdog after checkpointing -> run status "interrupted"


class VlmPipelineError(Exception):
    """Base class for every error raised by vlm_pipeline."""

    code: ClassVar[str] = "SYS-000"

    def __init__(self, message: str = "", *, params: Mapping[str, object] | None = None) -> None:
        super().__init__(message or self.code)
        self.params: dict[str, object] = dict(params or {})


class InsufficientMemoryError(VlmPipelineError):
    """Not enough unified memory to start a job (preflight refused)."""

    code: ClassVar[str] = "MEM-001"


# ------------------------------------------------------------------------------------------------ catalog

Severity = Literal["error", "warn", "info"]
ActionKind = Literal["patch", "resume", "navigate"]
_CODE_RE = re.compile(r"[A-Z]+-\d{3}")
_SEVERITIES = frozenset({"error", "warn", "info"})
_KINDS = frozenset({"patch", "resume", "navigate"})


@dataclass(frozen=True)
class Param:
    name: str
    type: str = "any"
    values: tuple[object, ...] = ()


@dataclass(frozen=True)
class Action:
    id: str
    kind: ActionKind
    confirm: bool
    target: str | None = None
    patch: tuple[Mapping[str, object], ...] | None = None
    when: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    when: Mapping[str, object]
    key: str


@dataclass(frozen=True)
class Entry:
    code: str
    slug: str
    severity: Severity
    prevent_in_preflight: bool
    params: tuple[Param, ...]
    actions: tuple[Action, ...]
    match: tuple[Mapping[str, str], ...] = ()
    variant_by: str | None = None
    variants: tuple[Variant, ...] = ()
    deprecated: bool = False


class CatalogError(VlmPipelineError):
    """The error catalog file itself is malformed (developer error, caught by unit tests)."""


def _parse_param(raw: object, code: str) -> Param:
    if isinstance(raw, str):
        return Param(name=raw)
    if isinstance(raw, dict) and isinstance(raw.get("name"), str):
        ptype = str(raw.get("type", "any"))
        values = tuple(raw.get("values", ()))
        if ptype == "enum" and not values:
            raise CatalogError(f"{code}: enum param {raw['name']!r} needs values")
        return Param(name=raw["name"], type=ptype, values=values)
    raise CatalogError(f"{code}: bad param {raw!r}")


def _parse_entry(code: str, raw: Mapping[str, Any]) -> Entry:
    if not _CODE_RE.fullmatch(code):
        raise CatalogError(f"bad code {code!r}")
    severity = raw.get("severity")
    if severity not in _SEVERITIES:
        raise CatalogError(f"{code}: bad severity {severity!r}")
    params = tuple(_parse_param(p, code) for p in raw.get("params") or ())
    names = {p.name for p in params}
    actions: list[Action] = []
    for a in raw.get("actions") or ():
        if a.get("kind") not in _KINDS:
            raise CatalogError(f"{code}: action {a.get('id')!r} has bad kind {a.get('kind')!r}")
        when = dict(a.get("when") or {})
        if unknown := set(when) - names:
            raise CatalogError(f"{code}: action {a['id']!r} `when` uses undeclared params {sorted(unknown)}")
        patch = a.get("patch")
        actions.append(
            Action(
                id=str(a["id"]),
                kind=a["kind"],
                confirm=bool(a.get("confirm", False)),
                target=a.get("target"),
                patch=tuple(patch) if patch is not None else None,
                when=when,
            )
        )
    if severity == "error" and not actions:
        raise CatalogError(f"{code}: severity=error needs at least one action")
    variant_by = raw.get("variant_by")
    if variant_by is not None and variant_by not in names:
        raise CatalogError(f"{code}: variant_by {variant_by!r} is not a declared param")
    variants = tuple(Variant(when=dict(v["when"]), key=str(v["key"])) for v in raw.get("variants") or ())
    for v in variants:
        if unknown := set(v.when) - names:
            raise CatalogError(f"{code}: variant {v.key!r} uses undeclared params {sorted(unknown)}")
    if variant_by and variants:
        raise CatalogError(f"{code}: use either variant_by or variants, not both")
    return Entry(
        code=code,
        slug=str(raw["slug"]),
        severity=severity,
        prevent_in_preflight=bool(raw.get("prevent_in_preflight", False)),
        params=params,
        actions=tuple(actions),
        match=tuple(raw.get("match") or ()),
        variant_by=variant_by,
        variants=variants,
        deprecated=bool(raw.get("deprecated", False)),
    )


@functools.cache
def load_catalog() -> dict[str, Entry]:
    """Load and validate ``common/error_catalog.yaml`` (cached)."""
    # Lazy: ExitCode / MemGuard users (container workers run via PYTHONPATH) must stay stdlib-only.
    import yaml  # noqa: PLC0415

    text = resources.files("vlm_pipeline.common").joinpath("error_catalog.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise CatalogError("error_catalog.yaml must be a mapping of code -> entry")
    catalog = {str(code): _parse_entry(str(code), entry) for code, entry in raw.items()}
    if "SYS-000" not in catalog:
        raise CatalogError("fallback SYS-000 missing")
    slugs = [e.slug for e in catalog.values()]
    if len(slugs) != len(set(slugs)):
        raise CatalogError("duplicate slug")
    return catalog


class _ErrorLike(Protocol):
    code: str
    params: Mapping[str, object]


def _matches(when: Mapping[str, object], env: Mapping[str, object]) -> bool:
    for name, expected in when.items():
        allowed = expected if isinstance(expected, list) else [expected]
        if env.get(name) not in allowed:
            return False
    return True


def available_actions(error: _ErrorLike) -> list[str]:
    """Catalog action ids offered for ``error`` (filtered by each action's ``when``).

    ``resumable`` comes from the error object (set by the executor from a real checkpoint) and defaults to False,
    so ``resume_checkpoint`` is never offered without one.
    """
    entry = load_catalog().get(error.code) or load_catalog()["SYS-000"]
    env: dict[str, object] = {**error.params, "resumable": bool(getattr(error, "resumable", False))}
    return [a.id for a in entry.actions if _matches(a.when, env)]
