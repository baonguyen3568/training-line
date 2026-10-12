"""Spec tests for the pre-run graph validator (P009 §4, P003 D11).

ASSUMED INTERFACE (draft — agree with bao-backend, rename both sides together):
- ``vlm_pipeline.graph.validate(workflow: dict, registry, workspace: Path | None = None) -> list[Issue]`` — never
  raises on bad input; pure on the dict, and when ``workspace`` is given also resolves ``LocalRef`` paths on disk
  (realpath must stay inside ``workspace`` — symlink escape).
- ``Issue = {node_id, port, param, code, severity: "error"|"warn", params}`` (agreed with bao-backend). No text in
  Issue: text is resolved from i18n keys ``error.<code>.title|what|why`` at the API layer
  (``src/vlm_pipeline/i18n/{vi,en}.yaml``). ASSUMED ``vlm_pipeline.i18n.load(locale) -> dict[str, str]`` (flat keys).
- ``vlm_pipeline.graph.testing.make_test_registry()`` returns a registry with CPU-only dummy node types
  (every param used by a test is DECLARED, so a test can never pass on "unknown param" by accident):
  ``test.dataset`` (out ``data: Dataset``; params ``path: LocalRef path = "data"``, ``note: str = ""``),
  ``test.model`` (out ``model: Model``; param ``revision: 40-hex str = "0"*40``),
  ``test.quant`` (in ``model: Model``; out ``model: QuantizedModel``),
  ``test.train`` (in ``data: Dataset``, ``base_model: Model``; out ``adapter: Adapter``;
  params ``max_steps: int 1..100000 = 100``, ``lr: float > 0 = 1e-4``),
  ``test.eval`` (in ``model: Model``; out ``report: EvalReport``).
Codes follow ``src/vlm_pipeline/common/error_catalog.yaml`` (owner bao-fixcode).
CFG-001..007, SEC-001/002 approved by bao-fixcode;
CFG-008 (duplicate node id) proposed.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.xfail(
    strict=True, raises=ImportError, reason="spec: bao-backend implements vlm_pipeline.graph (P009) in P2"
)


def _wf(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


VALID = _wf(
    [_n("d", "test.dataset"), _n("m", "test.model"), _n("t", "test.train", max_steps=10)],
    [_e("d", "data", "t", "data"), _e("m", "model", "t", "base_model")],
)


def _codes(issues: list[Any]) -> set[str]:
    return {i.code for i in issues if i.severity == "error"}


def _run(wf: dict[str, Any]) -> list[Any]:
    from vlm_pipeline.graph import validate
    from vlm_pipeline.graph.testing import make_test_registry

    return validate(wf, make_test_registry())


def test_valid_graph_has_no_errors() -> None:
    assert _codes(_run(VALID)) == set()


def test_cycle_is_rejected() -> None:
    wf = _wf(
        [_n("m", "test.model"), _n("q1", "test.quant"), _n("q2", "test.quant")],
        [_e("q1", "model", "q2", "model"), _e("q2", "model", "q1", "model")],
    )
    assert "CFG-002" in _codes(_run(wf))


def test_missing_required_input_points_at_node() -> None:
    wf = copy.deepcopy(VALID)
    wf["edges"] = [_e("d", "data", "t", "data")]  # base_model left unconnected
    issues = [i for i in _run(wf) if i.code == "CFG-003"]
    assert issues
    assert issues[0].node_id == "t"


def test_type_mismatch_is_rejected() -> None:
    wf = _wf([_n("d", "test.dataset"), _n("e", "test.eval")], [_e("d", "data", "e", "model")])
    assert "CFG-001" in _codes(_run(wf))


def test_subtype_connects_to_supertype() -> None:
    wf = _wf(
        [_n("m", "test.model"), _n("q", "test.quant"), _n("e", "test.eval")],
        [_e("m", "model", "q", "model"), _e("q", "model", "e", "model")],
    )
    assert _codes(_run(wf)) == set()


def test_unknown_node_type_is_rejected_not_installed() -> None:
    wf = _wf([_n("x", "evil.install_me")], [])
    assert "CFG-004" in _codes(_run(wf))


def test_param_out_of_range() -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"][2]["params"]["max_steps"] = 0
    assert "CFG-005" in _codes(_run(wf))


def test_edge_to_nonexistent_node() -> None:
    wf = copy.deepcopy(VALID)
    wf["edges"].append(_e("ghost", "data", "t", "data"))
    assert "CFG-006" in _codes(_run(wf))


# ---- P003 D11: workflow file is pure data ------------------------------------------------------
def test_unknown_top_level_field_rejected() -> None:
    wf = copy.deepcopy(VALID)
    wf["post_run_hook"] = "rm -rf /"
    assert "CFG-007" in _codes(_run(wf))


@pytest.mark.parametrize("path", ["/etc/passwd", "../../outside", "data/../../x"])
def test_paths_must_stay_inside_workspace(path: str) -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"][0]["params"]["path"] = path
    assert "SEC-001" in _codes(_run(wf))


@pytest.mark.parametrize("secret", ["hf_" + "a" * 34, "nvapi-" + "b" * 40, "sk-" + "c" * 40])
def test_secret_values_are_rejected(secret: str) -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"][0]["params"]["note"] = secret
    assert "SEC-002" in _codes(_run(wf))


def test_validator_never_raises_on_garbage() -> None:
    from vlm_pipeline.graph import validate
    from vlm_pipeline.graph.testing import make_test_registry

    garbage_inputs: list[Any] = [
        {},
        {"nodes": "x"},
        {"format": "vlm-workflow", "schema_version": 999},
        {"nodes": [None]},
        {"nodes": [{"id": "a"}], "edges": [{"from": "x"}]},
        None,
        [],
    ]
    for garbage in garbage_inputs:
        issues = validate(garbage, make_test_registry())
        assert issues, garbage


def test_every_issue_code_is_in_catalog_with_i18n_text() -> None:
    """No-tech UX: every Issue resolves to catalog text in both locales (display rules: test_error_catalog_spec)."""
    from vlm_pipeline.common.error_catalog import load_catalog
    from vlm_pipeline.i18n import load

    catalog, vi, en = load_catalog(), load("vi"), load("en")
    wf = _wf([_n("d", "test.dataset"), _n("e", "test.eval")], [_e("d", "data", "e", "model")])
    issues = _run(wf)
    assert issues
    for issue in issues:
        assert re.fullmatch(r"[A-Z]+-\d{3}", issue.code)
        assert issue.code in catalog
        for part in ("title", "what", "why"):
            assert f"error.{issue.code}.{part}" in vi
            assert f"error.{issue.code}.{part}" in en


# ---- P003 D11 additions requested by bao-security ----------------------------------------------
@pytest.mark.parametrize("path", ["%2e%2e/%2e%2e/etc", "data/%2E%2E/%2E%2E/x", "..%2fsecret"])
def test_percent_encoded_traversal_rejected(path: str) -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"][0]["params"]["path"] = path
    assert "SEC-001" in _codes(_run(wf))


def test_symlink_escaping_workspace_rejected(tmp_path: Path) -> None:
    from vlm_pipeline.graph import validate
    from vlm_pipeline.graph.testing import make_test_registry

    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "data").symlink_to(outside, target_is_directory=True)
    wf = copy.deepcopy(VALID)
    wf["nodes"][0]["params"]["path"] = "data"
    issues = validate(wf, make_test_registry(), workspace=workspace)
    assert "SEC-001" in {i.code for i in issues if i.severity == "error"}


def test_secret_is_never_echoed_back() -> None:
    secret = "hf_" + "z" * 34
    wf = copy.deepcopy(VALID)
    wf["nodes"][0]["params"]["note"] = secret
    issues = [i for i in _run(wf) if i.code == "SEC-002"]
    assert issues
    for issue in issues:
        assert secret not in repr(issue)
        assert issue.param == "note"  # names the field only


def test_duplicate_node_ids_rejected() -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"].append(_n("d", "test.dataset"))
    assert "CFG-008" in _codes(_run(wf))  # proposed code: duplicate node id


@pytest.mark.parametrize("field", ["install_url", "source", "pip", "git"])
def test_workflow_cannot_carry_install_sources(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """P003 D16 / P009 §2.4: a workflow file never installs anything and never triggers a network request."""
    import socket

    def _no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("validator must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    wf = copy.deepcopy(VALID)
    wf["nodes"].append({**_n("x", "acme/ocr-tools.read"), field: "https://evil.example/pack.tar.gz"})
    codes = _codes(_run(wf))
    assert "CFG-007" in codes  # extra=forbid on node objects
    assert "CFG-004" in codes  # node type not installed, reported, never auto-installed


def test_package_hint_from_workflow_is_never_trusted() -> None:
    """P003 D16: a package name written in a workflow is only a registry lookup key. A spoofed first-party name
    unknown to the registry yields ``suggested_package=None`` and the string is never echoed back.
    Node field ``package`` (optional, registry lookup key only) - confirmed by bao-backend (P009 §2.4.1).
    CFG-004 params: {node_type, reason: not_installed|not_allowed, trust_level, name_lookalike_of,
    suggested_package}."""
    hint = "vlm/lora-trainer"
    wf = copy.deepcopy(VALID)
    wf["nodes"].append({**_n("x", "acme/fake.train"), "package": hint})
    issues = [i for i in _run(wf) if i.code == "CFG-004"]
    assert issues
    issue = issues[0]
    assert issue.params["reason"] == "not_installed"
    assert issue.params["trust_level"] == "unverified"
    assert issue.params["suggested_package"] is None
    assert hint not in repr(issue)


@pytest.mark.parametrize(
    "type_id", ["acme/ocr-t\u043e\u043els.read", "Acme/ocr.read", "acme/ocr\u200b.read", "acme/\u202eocr.read"]
)
def test_node_type_charset_enforced(type_id: str) -> None:
    wf = copy.deepcopy(VALID)
    wf["nodes"].append(_n("x", type_id))
    assert "CFG-007" in _codes(_run(wf))
