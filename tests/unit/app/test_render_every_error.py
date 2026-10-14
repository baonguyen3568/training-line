"""Every catalog code and variant renders through the API's own path (E2E 2026-10-14: CFG-005 with a ``key`` param
made POST /api/validate answer 500, ``t() got multiple values for argument 'key'``)."""

from __future__ import annotations

from typing import Any

import pytest

from vlm_pipeline.app.main import _render_issue
from vlm_pipeline.common.error_catalog import load_catalog
from vlm_pipeline.i18n import LOCALES

CATALOG = load_catalog()


def _cases() -> list[tuple[str, dict[str, Any]]]:
    cases = []
    for code, entry in sorted(CATALOG.items()):
        base = {
            p.name: (p.values[0] if p.values else True if p.type == "bool" else f"<{p.name}>") for p in entry.params
        }
        cases.append((code, base))
        cases += [(code, {**base, **variant.when}) for variant in entry.variants]
    return cases


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize(("code", "params"), _cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_error_renders(code: str, params: dict[str, Any], locale: str) -> None:
    out = _render_issue({"code": code, "params": params, "node_id": "n1"}, locale)
    assert out["title"]
    assert out["title"] != f"error.{code}.title"  # a real text, not the key
    assert out["what"]
    assert not str(out["what"]).startswith("error.")


def test_a_param_named_key_or_locale_reaches_the_text() -> None:
    out = _render_issue({"code": "CFG-005", "params": {"node": "n", "param": "advanced", "value": 1, "rule": "type",
                                                       "key": "optimizer.eps", "locale": "x"}}, "en")  # fmt: skip
    assert "optimizer.eps" in out["what"]
