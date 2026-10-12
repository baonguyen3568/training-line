"""Tests for the model-vetting helper scripts/research/hf_model_meta.py (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "research" / "hf_model_meta.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hf_model_meta", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hf = _load()


def test_root_auto_map_to_other_repo_is_reported() -> None:
    cfg = {"auto_map": {"AutoModel": "evil-org/other-repo--modeling.Cls", "AutoConfig": "configuration.Cfg"}}
    out: dict[str, Any] = {}
    hf.collect_bridges(cfg, "", out)
    assert out == {"auto_map": ["evil-org/other-repo"]}


def test_root_auto_map_local_and_nested_bridges() -> None:
    cfg = {
        "auto_map": {"AutoModel": "modeling.Cls"},
        "encoder": {"_name_or_path": "nvidia/C-RADIOv2-H", "auto_map": {"AutoModel": "nvidia/C-RADIOv2-H--hf_model.M"}},
    }
    out: dict[str, Any] = {}
    hf.collect_bridges(cfg, "", out)
    assert out == {
        "auto_map": "local",
        "encoder._name_or_path": "nvidia/C-RADIOv2-H",
        "encoder.auto_map": ["nvidia/C-RADIOv2-H"],
    }


@pytest.mark.parametrize("repo_id", ["../etc", "org/..", "./x", "org/name?x=1", "org", "org/na/me", "org/name#frag"])
def test_invalid_repo_id_rejected_without_network(repo_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "fetch_json", lambda url: pytest.fail(f"network called: {url}"))
    _, ok = hf.describe(repo_id)
    assert ok is False


def test_main_exits_nonzero_when_any_repo_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(url: str) -> dict[str, Any]:
        if "/api/models/good/" in url:
            return {"sha": "a" * 40, "siblings": [], "tags": []}
        if "/good/" in url:
            return {"model_type": "x"}
        return {"_err": "HTTP Error 404"}

    monkeypatch.setattr(hf, "fetch_json", fake)
    assert hf.main(["good/model"]) == 0
    assert hf.main(["good/model", "bad/model"]) == 1
    assert hf.main([]) == 2


def test_missing_sha_is_an_error_not_keyerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "fetch_json", lambda url: {"siblings": []})
    report, ok = hf.describe("org/name")
    assert ok is False
    assert "missing sha" in report


@pytest.mark.parametrize("repo_id", ["nvidia/NVIDIA-Nemotron-Parse-v1.1", "Qwen/Qwen3.5-0.8B", "google/gemma-4-E2B-it"])
def test_real_repo_ids_accepted(repo_id: str) -> None:
    assert hf.REPO_ID_RE.match(repo_id)
