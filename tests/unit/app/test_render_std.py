"""STD-001 (P4-2 standards): the issue's own standards.* sentence replaces the generic text; nothing else can."""

from __future__ import annotations

import pytest

from vlm_pipeline import i18n
from vlm_pipeline.app.main import _render_issue


def test_std001_what_comes_from_its_standards_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_t = i18n.t

    def fake_t(key: str, locale: str, **params: object) -> str:
        if key == "standards.recipe.lr_too_high":
            return f"LR {params['value']} > {params['max']}"
        return real_t(key, locale, **params)

    monkeypatch.setattr(i18n, "t", fake_t)
    issue = {"code": "STD-001", "node_id": "t", "params": {"node": "t", "message": "standards.recipe.lr_too_high",
                                                          "rule": "lr", "value": 0.01, "max": 0.001}}  # fmt: skip
    assert _render_issue(issue, "vi")["what"] == "LR 0.01 > 0.001"


@pytest.mark.parametrize("message", ["error.SYS-000.title", "standards.", "standards/../x", "", None])
def test_other_keys_never_replace_the_generic_text(message: object) -> None:
    issue = {"code": "STD-001", "params": {"node": "t", "message": message}}
    assert _render_issue(issue, "vi")["what"] == i18n.t("error.STD-001.what", "vi", node="t", message=message)
