"""Unit tests for the error catalog loader (text checks live in tests/unit/test_error_catalog_spec.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlm_pipeline.common import error_catalog
from vlm_pipeline.common.error_catalog import CatalogError, Entry, available_actions, load_catalog
from vlm_pipeline.common.errors import ExitCode, InsufficientMemoryError


def test_catalog_loads_and_has_fallback() -> None:
    catalog = load_catalog()
    assert "SYS-000" in catalog
    assert len({e.slug for e in catalog.values()}) == len(catalog)


def test_every_error_has_an_action_and_destructive_ops_never_patch() -> None:
    for code, entry in load_catalog().items():
        if entry.severity == "error":
            assert entry.actions, code
        for action in entry.actions:
            if action.id in {"free_space", "review_to_install", "show_admin_command"}:
                assert action.kind == "navigate", (code, action.id)


def test_resume_checkpoint_always_guarded_by_resumable() -> None:
    for code, entry in load_catalog().items():
        for action in entry.actions:
            if action.id in {"resume_checkpoint", "resume_lighter"}:
                assert action.when == {"resumable": True}, code


@pytest.mark.parametrize("code", ["MEM-003", "SYS-003", "SYS-000", "MEM-002", "DISK-001", "NET-001"])
def test_resume_offered_iff_resumable(code: str) -> None:
    yes = available_actions(SimpleNamespace(code=code, params={}, resumable=True))
    no = available_actions(SimpleNamespace(code=code, params={}, resumable=False))
    assert "resume_checkpoint" in yes
    assert "resume_checkpoint" not in no


def test_missing_resumable_defaults_to_false() -> None:
    assert "resume_checkpoint" not in available_actions(SimpleNamespace(code="SYS-000", params={}))


@pytest.mark.parametrize(
    ("params", "offered"),
    [({"reason": "not_installed"}, True), ({"reason": "not_allowed"}, False)],
)
def test_cfg004_install_only_when_not_installed(params: dict[str, str], offered: bool) -> None:
    assert ("review_to_install" in available_actions(SimpleNamespace(code="CFG-004", params=params))) is offered


def test_unknown_code_falls_back_to_sys000_actions() -> None:
    assert available_actions(SimpleNamespace(code="NOPE-999", params={})) == available_actions(
        SimpleNamespace(code="SYS-000", params={})
    )


def test_exception_carries_code_and_params_only() -> None:
    err = InsufficientMemoryError(params={"need_gb": 52.0, "avail_gb": 43.2})
    assert err.code == "MEM-001"
    assert err.code in load_catalog()
    assert err.params == {"need_gb": 52.0, "avail_gb": 43.2}


def test_exit_codes_are_stable() -> None:
    assert [int(c) for c in ExitCode] == [0, 2, 3, 4, 75]


def test_typo_in_entry_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="prevent_in_prefight"):
        Entry.model_validate(
            {"code": "X-001", "slug": "XX", "severity": "info", "prevent_in_preflight": True, "prevent_in_prefight": 1}
        )


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ({"severity": "error", "actions": []}, "at least one action"),
        (
            {"severity": "info", "actions": [{"id": "a", "kind": "patch", "confirm": False, "when": {"zz": True}}]},
            "undeclared",
        ),
        ({"severity": "info", "actions": [{"id": "a", "kind": "delete", "confirm": True}]}, "kind"),
        ({"severity": "info", "variant_by": "nope"}, "variant_by"),
        ({"severity": "info", "params": [{"name": "r", "type": "enum"}]}, "needs values"),
    ],
)
def test_cross_field_rules(body: dict[str, object], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        Entry.model_validate({"code": "X-001", "slug": "XX", "prevent_in_preflight": False, **body})


def test_malformed_yaml_raises_catalog_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(error_catalog.yaml, "safe_load", lambda _text: {"MEM-001": {"slug": "X", "severity": "boom"}})
    error_catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(CatalogError, match="MEM-001"):
            error_catalog.load_catalog()
    finally:
        error_catalog.load_catalog.cache_clear()
