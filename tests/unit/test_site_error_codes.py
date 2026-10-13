"""/docs/ma-loi reference (site/scripts/error_codes.py): complete, unique anchors, public text only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.error_catalog import load_catalog

SCRIPT = Path(__file__).resolve().parents[2] / "site" / "scripts" / "error_codes.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("site_error_codes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    return dict(_module().build())


def test_every_catalog_code_has_an_entry_with_a_unique_anchor(data: dict[str, Any]) -> None:
    codes = [c for g in data["groups"] for c in g["codes"]]
    anchors = [c["anchor"] for c in codes]
    assert len(anchors) == len(set(anchors))
    expected = {code for code, entry in load_catalog().items() if not entry.deprecated}
    assert {c["code"] for c in codes} == expected


def test_entries_carry_text_in_both_locales_and_nothing_internal(data: dict[str, Any]) -> None:
    for group in data["groups"]:
        assert group["title"]["vi"]
        assert group["title"]["en"]
        for c in group["codes"]:
            assert set(c) == {"code", "anchor", "severity", "title", "what", "actions", "guide"}
            assert c["what"]["vi"]
            assert len(c["what"]["vi"]) == len(c["what"]["en"]), c["code"]
    blob = json.dumps(data, ensure_ascii=False)
    for code, entry in load_catalog().items():  # log regexes / internal ids never reach the public page
        assert entry.slug not in blob, code
        for rule in entry.match:
            assert str(rule.get("value")) not in blob, (code, rule)


def test_variants_are_listed_and_memory_codes_link_the_guide(data: dict[str, Any]) -> None:
    codes = {c["code"]: c for g in data["groups"] for c in g["codes"]}
    assert len(codes["CFG-010"]["what"]["vi"]) == 3  # full_finetune_only / lora_only / not_trainable
    assert codes["MEM-001"]["guide"] == "/docs/xu-ly-su-co-bo-nho/#MEM-001"  # bao-cleanup's explicit anchor
    assert codes["SYS-001"]["guide"] == "/docs/xu-ly-su-co-bo-nho/#SYS-001"
    assert codes["CFG-001"]["guide"] is None


def test_a_prefix_without_group_text_fails_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    import vlm_pipeline.i18n as i18n

    real = i18n.load
    monkeypatch.setattr(i18n, "load", lambda loc: {k: v for k, v in real(loc).items() if k != "error_group.MEM"})
    with pytest.raises(SystemExit, match=r"error_group\.MEM"):
        _module().build()
