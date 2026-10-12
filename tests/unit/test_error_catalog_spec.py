"""Spec tests for the error catalog + its i18n text.

Single source of truth: ``src/vlm_pipeline/common/error_catalog.yaml`` holds codes/severity/action keys only;
user text lives in ``src/vlm_pipeline/i18n/{vi,en}.yaml`` under ``error.<code>.title|what|why`` (bao-design/backend).
Display rules: P010 §6–§7 + ``.team/knowledge/error-catalog-draft.md`` ("Quyết định hiển thị", bao-design):
title ≤ 10 tokens in vi (``split()`` counts syllables) / ≤ 8 words in en, explanation ≤ 2 sentences, ≥ 1 action, no traceback / secret / $HOME path in user text.
Catalog entry shape (owner bao-fixcode): ``{code, slug, severity, match[], prevent_in_preflight,
actions[{id, kind, target, patch, confirm}], params[]}`` — no text. ``slug`` is engineer-only and not checked.
ASSUMED: ``vlm_pipeline.common.errors.load_catalog() -> dict[code, Entry]`` (attribute access); ``vlm_pipeline.i18n.load(locale) -> dict[str, str]`` (flat keys).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-fixcode/bao-backend implement common/errors.py + i18n in P2"
)

LOCALES = ("vi", "en")
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}")


def _load() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    from vlm_pipeline.common.errors import load_catalog
    from vlm_pipeline.i18n import load

    return load_catalog(), {loc: load(loc) for loc in LOCALES}


def test_codes_are_well_formed_and_fallback_exists() -> None:
    catalog, _ = _load()
    assert "SYS-000" in catalog
    for code in catalog:
        assert re.fullmatch(r"[A-Z]+-\d{3}", code), code


def test_every_code_has_text_in_every_locale() -> None:
    catalog, texts = _load()
    for code, entry in catalog.items():
        assert entry.severity in {"error", "warn", "info"}, code
        assert entry.actions, code
        for loc, t in texts.items():
            for part in ("title", "what", "why"):
                assert t.get(f"error.{code}.{part}", "").strip(), (loc, code, part)
            for action in entry.actions:
                assert t.get(f"action.{action.id}", "").strip(), (loc, code, action.id)


def test_display_rules() -> None:
    catalog, texts = _load()
    for code in catalog:
        for loc, t in texts.items():
            title = t[f"error.{code}.title"]
            max_tokens = {"vi": 10, "en": 8}[loc]
            assert 0 < len(title.split()) <= max_tokens, (loc, code, title)
            for part in ("what", "why"):
                text = t[f"error.{code}.{part}"].strip()
                assert len(re.findall(r"[.!?](\s|$)", text)) <= 2, (loc, code, part)


def test_no_internals_in_user_text() -> None:
    _, texts = _load()
    for loc, t in texts.items():
        for key, text in t.items():
            if not key.startswith("error."):
                continue
            assert "Traceback" not in text, (loc, key)
            assert "/home/" not in text and "~/" not in text, (loc, key)
            assert not SECRET_RE.search(text), (loc, key)


def test_declared_params_cover_placeholders() -> None:
    """Every ``{x}`` used in an error text must be declared in the catalog entry's ``params``."""
    catalog, texts = _load()
    for code, entry in catalog.items():
        for loc, t in texts.items():
            for part in ("title", "what", "why"):
                used = set(re.findall(r"\{(\w+)\}", t[f"error.{code}.{part}"]))
                assert used <= set(entry.params), (loc, code, part, used - set(entry.params))


def test_placeholders_match_across_locales() -> None:
    """``{need}``-style placeholders must be identical in vi and en, so rendering never shows a raw ``{x}``."""
    _, texts = _load()
    vi, en = texts["vi"], texts["en"]
    for key in vi.keys() & en.keys():
        assert set(re.findall(r"\{(\w+)\}", vi[key])) == set(re.findall(r"\{(\w+)\}", en[key])), key


def test_locales_have_same_keys() -> None:
    _, texts = _load()
    assert texts["vi"].keys() == texts["en"].keys()
