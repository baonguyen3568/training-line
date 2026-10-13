"""Usability-test fixtures (tests/fixtures/usability) against the real registry: expected_issues.json is the oracle.

A fixture that drifts from the node registry would put a different error (or a stack of CFG-005) in front of the
participant than the task script expects, and the session would test the wrong thing. Issues coming from preflight
(``via`` a stub, e.g. MEM-001) are not produced by graph validation: those workflows must validate clean.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app.main import _labels, _render_issue
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "usability"
EXPECTED: dict[str, list[dict[str, Any]]] = json.loads((FIXTURES / "expected_issues.json").read_text())


def _workflow(name: str) -> dict[str, Any]:
    return dict(json.loads((FIXTURES / name).read_text()))


def _validated(name: str) -> list[Any]:
    return list(validate(_workflow(name), REGISTRY))


def _what(issue: Any, name: str) -> str:
    """The vi ``what`` text the participant reads, rendered by the same code as /api/validate."""
    labels = _labels(_workflow(name), REGISTRY, "vi")
    return str(_render_issue(asdict(issue), "vi", labels)["what"])


def test_every_fixture_has_an_oracle_entry() -> None:
    assert sorted(p.name for p in FIXTURES.glob("*.vlmflow.json")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_raises_exactly_the_expected_issues(name: str) -> None:
    expected = [e for e in EXPECTED[name] if "via" not in e]
    issues = _validated(name)
    assert [(i.code, i.node_id) for i in issues] == [(e["code"], e["node_id"]) for e in expected]
    for issue, exp in zip(issues, expected, strict=True):
        if "port" in exp:
            assert issue.port == exp["port"]
        for param, value in exp.get("params", {}).items():
            assert issue.params[param] == value, (issue.code, param)
        text = _what(issue, name)
        for forbidden in exp.get("must_not_contain", []):
            assert forbidden not in text, (issue.code, text)


def test_parse_on_lora_tells_the_user_what_to_do() -> None:
    [issue] = _validated("t6_parse_on_lora.vlmflow.json")
    text = _what(issue, "t6_parse_on_lora.vlmflow.json")
    assert "NVIDIA Nemotron Parse v1.1" in text
    assert "fine-tune toàn phần" in text
    assert "{" not in text  # every placeholder filled
