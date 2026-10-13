"""Every licence duty and verdict the catalog can send has copy in every locale (design 4fd5f71).

The download dialog hides a duty without copy as a safety net, so a missing key would silently drop a legal duty
from the consent text the user agrees to.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vlm_pipeline import i18n

OBLIGATIONS = Path(__file__).resolve().parents[3] / "configs" / "licenses" / "obligations.yaml"
GROUP_KEYS = ("obligations_title", "obligations_use", "obligations_distribute", "policy_link")


def _licences() -> dict[str, dict[str, object]]:
    return yaml.safe_load(OBLIGATIONS.read_text(encoding="utf-8"))["licences"]


@pytest.mark.parametrize("locale", i18n.LOCALES)
def test_every_obligation_and_class_has_copy(locale: str) -> None:
    flat = i18n.load(locale)
    licences = _licences()
    ids = {o["id"] for lic in licences.values() for o in lic.get("obligations") or []}  # type: ignore[union-attr]
    classes = {str(lic["license_class"]) for lic in licences.values()}
    assert sorted(i for i in ids if f"license.obligation.{i}" not in flat) == []
    assert sorted(c for c in classes if f"license.badge.{c}" not in flat) == []
    assert [k for k in GROUP_KEYS if f"license.{k}" not in flat] == []
