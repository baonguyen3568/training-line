"""vLLM <-> HF parity gate: the four conditions decide; ids pair by image; near ties are reported."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "vllm_hf_parity.py"
GT = {"total": {"total_price": "10.000"}}


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("vllm_hf_parity", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(n: int, *, differ: int = 0, sha_differ: int = 0) -> tuple[dict[int, Any], dict[int, Any]]:
    good, bad = json.dumps(GT), json.dumps({"total": {"total_price": "99"}})
    hf, vl = {}, {}
    for i in range(n):
        hf[i] = {"sequence": good, "gt_parse": GT, "truncated": False, "prompt_ids_sha": "s", "image_sha256": "img",
                 "token_ids": [1, 2, 3, 7]}  # fmt: skip
        diff = i < differ
        vl[i] = {
            "text": bad if diff else good,
            "finish_reason": "stop",
            "prompt_ids_sha": "x" if i < sha_differ else "s",
            "image_sha256": "img",
            "token_ids": [1, 9, 3, 7] if diff else [1, 2, 3, 7],  # 7 = EOS on a "stop" finish
            "top_logprobs": [[[1, -0.01], [5, -5.0]], [[9, -0.70], [2, -0.75]], [[3, -0.01], [4, -6.0]]],
        }
    return hf, vl


def test_identical_outputs_pass() -> None:
    report = _mod().compare(*_pair(50))
    assert report["verdict"] == "PASS"
    assert report["exact_match_pct"] == 100.0


def test_a_different_prompt_alone_fails_the_gate() -> None:
    report = _mod().compare(*_pair(50, sha_differ=1))
    assert report["checks"]["1_same_input"] is False
    assert report["verdict"] == "FAIL"


def test_divergences_report_the_first_token_and_the_near_tie() -> None:
    report = _mod().compare(*_pair(50, differ=3))
    assert report["exact_match_pct"] == 94.0
    assert report["diverged"][0] == {
        "id": 0,
        "first_diff_token": 1,
        "hf_choice_rank_in_vllm": "2",
        "vllm_margin_nats": 0.05,
    }
    assert report["near_ties"] == 3


def test_too_many_differences_fail_exact_and_f1() -> None:
    report = _mod().compare(*_pair(50, differ=10))
    assert report["checks"]["2_exact_match_ge_90"] is False
    assert report["checks"]["3_f1_delta_le_2_and_ci_has_0"] is False


def test_images_must_be_the_same() -> None:
    hf, vl = _pair(5)
    del vl[4]
    with pytest.raises(SystemExit, match="different images"):
        _mod().compare(hf, vl)


def test_a_different_image_or_an_image_error_fails_same_input() -> None:
    hf, vl = _pair(10)
    hf[0]["image_sha256"] = "other"
    vl[1] = {"id": "1", "error": {"code": "IMAGE", "reason": "too_many_pixels"}}
    report = _mod().compare(hf, vl)
    assert report["checks"]["1_same_input"] is False
    assert report["vllm_image_errors"] == [1]


def test_an_hf_token_outside_vllm_top2_is_not_a_near_tie() -> None:
    hf, vl = _pair(10, differ=1)
    hf[0]["token_ids"] = [1, 8, 3, 7]  # 8 is in neither of vLLM's top-2 at position 1
    vl[0]["text"] = "x"
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["hf_choice_rank_in_vllm"] == ">2"
    assert report["near_ties"] == 0


def test_no_predictions_is_an_error() -> None:
    with pytest.raises(SystemExit, match="no HF predictions"):
        _mod().compare({}, {})
