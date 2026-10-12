"""Spec tests for the error catalog + its i18n text.

Single source of truth: ``src/vlm_pipeline/common/error_catalog.yaml`` holds codes/severity/action keys only;
user text lives in ``src/vlm_pipeline/i18n/{vi,en}.yaml`` under ``error.<code>.title|what`` + OPTIONAL ``why``
(bao-design P010 §7: ``why`` only for a run-specific cause computed from params;
if present: ≤ 1 sentence, both locales).
Display rules: P010 §6-§7 + ``.team/knowledge/error-catalog-draft.md`` ("Quyết định hiển thị", bao-design):
title ≤ 10 tokens in vi (``split()`` counts syllables) / ≤ 8 words in en, explanation ≤ 2 sentences,
≥ 1 action for severity=error, no placeholder in titles, no traceback / secret / $HOME path in user text.
Catalog entry shape (owner bao-fixcode): ``{code, slug, severity, match[], prevent_in_preflight,
actions[{id, kind, target, patch, confirm, when?}], params[], variant_by?}`` - no text. ``slug`` is engineer-only.
Text variants (bao-fixcode): ``variant_by: <param>`` (enum or bool) ⇒ i18n has ``error.<code>.what.<value>`` for EVERY
value (bool: ``<param>`` / ``not_<param>``) and NO bare ``what``; without ``variant_by`` a bare ``what`` is required.
Multi-param variants: ``variants: [{when: {param: value | [values]}, key}]`` evaluated in order; every combination
of the enum/bool params named in ``when`` must match >= 1 variant; keys live at ``error.<code>.what.<key>``.
``action.when: {<param>: <value>}`` ⇒ ``available_actions(err)`` offers the action only when params match.
Params are names or objects ``{name, type, values?}`` - helper ``_param_names`` accepts both.
ASSUMED: ``vlm_pipeline.common.error_catalog.load_catalog() -> dict[code, Entry]`` (attribute access);
``vlm_pipeline.i18n.load(locale) -> dict[str, str]`` (flat keys).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

# Catalog-only tests run for real (common/error_catalog.yaml + errors.load_catalog exist). Tests that need the
# i18n loader stay spec until ``vlm_pipeline.i18n.load`` lands (strict xfail, ImportError only).
NEEDS_I18N = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: vlm_pipeline.i18n.load not implemented yet (owner: bao-backend)"
)

LOCALES = ("vi", "en")
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}")


def _catalog() -> dict[str, Any]:
    from vlm_pipeline.common.error_catalog import load_catalog

    return load_catalog()


def _load() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    from vlm_pipeline.common.error_catalog import load_catalog
    from vlm_pipeline.i18n import load

    return load_catalog(), {loc: load(loc) for loc in LOCALES}


def _param(entry: Any, name: str) -> Any:
    for p in entry.params:
        if (p if isinstance(p, str) else p.name) == name:
            return p
    raise AssertionError(f"variant_by param {name!r} not declared")


def _param_names(entry: Any) -> set[str]:
    return {p if isinstance(p, str) else p.name for p in entry.params}


def _values(entry: Any, name: str) -> list[Any]:
    p = _param(entry, name)
    if p.type == "bool":
        return [True, False]
    assert p.type == "enum", (name, p.type)
    return list(p.values)


def _what_keys(code: str, entry: Any) -> list[str]:
    """All i18n keys holding the 'what' text of an entry (bare, one per variant_by value, or one per variants key)."""
    variants = getattr(entry, "variants", None)
    if variants:
        return [f"error.{code}.what.{v.key}" for v in variants]
    variant_by = getattr(entry, "variant_by", None)
    if not variant_by:
        return [f"error.{code}.what"]
    p = _param(entry, variant_by)
    if p.type == "bool":
        values = [variant_by, f"not_{variant_by}"]
    else:
        assert p.type == "enum", (code, p.type)
        values = list(p.values)
    return [f"error.{code}.what.{v}" for v in values]


def _text_keys(code: str, entry: Any, t: dict[str, str] | None = None) -> list[str]:
    """Required keys, plus ``why`` when the given locale defines it (``why`` is optional)."""
    keys = [f"error.{code}.title", *_what_keys(code, entry)]
    if t is not None and f"error.{code}.why" in t:
        keys.append(f"error.{code}.why")
    return keys


def test_codes_are_well_formed_and_fallback_exists() -> None:
    catalog = _catalog()
    assert "SYS-000" in catalog
    for code in catalog:
        assert re.fullmatch(r"[A-Z]+-\d{3}", code), code


@NEEDS_I18N
def test_every_code_has_text_in_every_locale() -> None:
    catalog, texts = _load()
    for code, entry in catalog.items():
        assert entry.severity in {"error", "warn", "info"}, code
        if entry.severity == "error":  # warn/info handled by the app itself (e.g. CACHE-001) may have no action
            assert entry.actions, code
        for loc, t in texts.items():
            for key in _text_keys(code, entry):
                assert t.get(key, "").strip(), (loc, key)
            if getattr(entry, "variant_by", None) or getattr(entry, "variants", None):
                assert f"error.{code}.what" not in t, (loc, code, "no bare 'what' when variant_by is set")
            for action in entry.actions:
                assert t.get(f"action.{action.id}", "").strip(), (loc, code, action.id)


@NEEDS_I18N
def test_display_rules() -> None:
    catalog, texts = _load()
    for code, entry in catalog.items():
        for loc, t in texts.items():
            title = t[f"error.{code}.title"]
            max_tokens = {"vi": 10, "en": 8}[loc]
            assert 0 < len(title.split()) <= max_tokens, (loc, code, title)
            assert not re.search(r"\{\w+\}", title), (loc, code, "no placeholder in titles")
            for key in _what_keys(code, entry):
                text = t[key].strip()
                assert len(re.findall(r"[.!?](\s|$)", text)) <= 2, (loc, key)
            why = t.get(f"error.{code}.why")
            if why is not None:
                assert why.strip(), (loc, code)
                assert len(re.findall(r"[.!?](\s|$)", why.strip())) <= 1, (loc, code, "why is one sentence")


@NEEDS_I18N
def test_no_internals_in_user_text() -> None:
    _, texts = _load()
    for loc, t in texts.items():
        for key, text in t.items():
            if not key.startswith("error."):
                continue
            assert "Traceback" not in text, (loc, key)
            assert "/home/" not in text, (loc, key)
            assert "~/" not in text, (loc, key)
            assert not SECRET_RE.search(text), (loc, key)


@NEEDS_I18N
def test_declared_params_cover_placeholders() -> None:
    """Every ``{x}`` used in an error text must be declared in the catalog entry's ``params``."""
    catalog, texts = _load()
    for code, entry in catalog.items():
        for loc, t in texts.items():
            for key in _text_keys(code, entry, t):
                used = set(re.findall(r"\{(\w+)\}", t[key]))
                assert used <= _param_names(entry), (loc, key, used - _param_names(entry))


@NEEDS_I18N
def test_placeholders_match_across_locales() -> None:
    """``{need}``-style placeholders must be identical in vi and en, so rendering never shows a raw ``{x}``."""
    _, texts = _load()
    vi, en = texts["vi"], texts["en"]
    for key in vi.keys() & en.keys():
        assert set(re.findall(r"\{(\w+)\}", vi[key])) == set(re.findall(r"\{(\w+)\}", en[key])), key


@NEEDS_I18N
def test_locales_have_same_keys() -> None:
    _, texts = _load()
    assert texts["vi"].keys() == texts["en"].keys()


def test_action_when_conditions_reference_declared_params() -> None:
    catalog = _catalog()
    for code, entry in catalog.items():
        for action in entry.actions:
            for param in getattr(action, "when", None) or {}:
                assert param in _param_names(entry), (code, action.id, param)


@pytest.mark.parametrize(
    ("code", "params", "action_id", "offered"),
    [
        (
            "CFG-004",
            {
                "node_type": "acme/x.y",
                "reason": "not_installed",
                "trust_level": "unverified",
                "suggested_package": None,
            },
            "review_to_install",
            True,
        ),
        (
            "CFG-004",
            {"node_type": "acme/x.y", "reason": "not_allowed", "trust_level": "unverified", "suggested_package": None},
            "review_to_install",
            False,
        ),
        ("CFG-009", {"migratable": True}, "migrate_node", True),
        ("CFG-009", {"migratable": False}, "migrate_node", False),
    ],
)
def test_available_actions_filters_by_when(code: str, params: dict[str, Any], action_id: str, offered: bool) -> None:
    from types import SimpleNamespace

    from vlm_pipeline.common.error_catalog import available_actions

    err = SimpleNamespace(code=code, params=params, resumable=False)
    assert (action_id in available_actions(err)) is offered


def test_variants_cover_every_combination() -> None:
    """Cartesian product of all enum/bool params used in ``variants[].when`` must each match >= 1 variant."""
    import itertools

    catalog = _catalog()
    for code, entry in catalog.items():
        variants = getattr(entry, "variants", None)
        if not variants:
            continue
        names = sorted({n for v in variants for n in v.when})
        for combo in itertools.product(*(_values(entry, n) for n in names)):
            env = dict(zip(names, combo, strict=True))

            def matches(variant: Any, env: dict[str, Any] = env) -> bool:
                for name, expected in variant.when.items():
                    allowed = expected if isinstance(expected, list) else [expected]
                    if env[name] not in allowed:
                        return False
                return True

            assert any(matches(v) for v in variants), (code, env)


@NEEDS_I18N
def test_why_is_optional_but_symmetric() -> None:
    """``why`` may be absent; if one locale has it, the other must too (covered for all keys by key parity,
    asserted here explicitly for the error cards)."""
    catalog, texts = _load()
    for code in catalog:
        has = {loc: f"error.{code}.why" in t for loc, t in texts.items()}
        assert len(set(has.values())) == 1, (code, has)
