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
        "gap_nats": 0.05,
        "kind": "near_tie",
    }
    assert report["near_ties"] == 3


def test_near_ties_alone_pass_2_and_3_but_worse_answers_fail_f1() -> None:
    report = _mod().compare(*_pair(50, differ=10))
    assert report["original_rule"]["2_exact_match_ge_90"] is False
    assert report["checks"]["2_no_anomaly"] is True
    assert report["checks"]["3_real_divergences_le_5pct"] is True  # 10 near ties, 0 real divergence...
    assert report["checks"]["4_f1_delta_le_2_and_ci_has_0"] is False  # ...but the answers are worse
    assert report["verdict"] == "FAIL"


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


def test_an_exact_tie_listed_first_is_a_near_tie() -> None:
    # First real probe (Qwen3.5-2B): vLLM lists HF's token first with the same logprob, then picks the other one.
    hf, vl = _pair(10, differ=1)
    vl[0]["top_logprobs"][1] = [[2, -0.69], [9, -0.69]]
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["hf_choice_rank_in_vllm"] == "1"
    assert report["diverged"][0]["vllm_margin_nats"] == 0.0
    assert report["near_ties"] == 1
    assert report["diverged"][0]["kind"] == "near_tie"


def test_hf_token_scored_above_vllm_choice_is_an_anomaly_and_fails() -> None:
    # vLLM greedy did not take its best token: not a tie, whatever else holds.
    hf, vl = _pair(50, differ=1)
    vl[0]["top_logprobs"][1] = [[2, -0.4], [9, -0.7]]
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["kind"] == "anomaly"
    assert report["diverged"][0]["gap_nats"] == -0.3
    assert report["anomalies"] == [0]
    assert report["checks"]["2_no_anomaly"] is False
    assert report["verdict"] == "FAIL"


def test_vllm_choice_missing_from_its_own_top_k_is_an_anomaly() -> None:
    hf, vl = _pair(10, differ=1)
    vl[0]["top_logprobs"][1] = [[2, -0.7], [5, -0.9]]  # vLLM generated 9, which its top-k does not list
    assert _mod().compare(hf, vl)["diverged"][0]["kind"] == "anomaly"


def test_real_divergences_over_5pct_fail_and_gap_decides_not_rank() -> None:
    hf, vl = _pair(10, differ=2)
    hf[0]["token_ids"] = [1, 8, 3, 7]  # outside vLLM's recorded top-k
    vl[1]["top_logprobs"][1] = [[9, -0.70], [2, -1.20]]  # rank 2 but gap 0.5: not < 0.5
    report = _mod().compare(hf, vl)
    assert report["real_divergences"] == [0, 1]
    assert report["real_divergence_pct"] == 20.0
    assert report["checks"]["3_real_divergences_le_5pct"] is False


def test_a_rank_3_token_within_half_a_nat_is_a_near_tie() -> None:
    hf, vl = _pair(10, differ=1)
    vl[0]["top_logprobs"][1] = [[9, -1.10], [4, -1.15], [2, -1.30]]  # flat: HF's token is third, gap 0.2
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["hf_choice_rank_in_vllm"] == "3"
    assert report["diverged"][0]["kind"] == "near_tie"


def test_a_base_answering_its_own_schema_is_not_quality_informative() -> None:
    hf, vl = _pair(20)
    for side, key in ((hf, "sequence"), (vl, "text")):
        for row in side.values():
            row[key] = json.dumps({"ticket": {"price": 1}})  # valid JSON, not CORD's schema: field F1 0
    report = _mod().compare(hf, vl)
    assert all(report["checks"].values())
    assert report["quality_informative"] is False
    assert report["verdict"] == "FAIL"  # a base alone never decides


def test_gate_needs_the_base_engine_checks_and_the_fine_tune_quality() -> None:
    mod = _mod()
    base_hf, base_vl = _pair(20)
    for side, key in ((base_hf, "sequence"), (base_vl, "text")):
        for row in side.values():
            row[key] = json.dumps({"ticket": {"price": 1}})
    base = mod.compare(base_hf, base_vl)
    tuned = mod.compare(*_pair(20))
    assert mod.gate(base, tuned)["verdict"] == "PASS"
    assert mod.gate(base, base)["failed"] == ["fine_tune:quality_informative"]
    bad_hf, bad_vl = _pair(20, differ=2)
    bad_hf[0]["token_ids"] = [1, 8, 3, 7]
    bad_hf[1]["token_ids"] = [1, 8, 3, 7]
    worse = mod.compare(bad_hf, bad_vl)  # 2/20 real divergences = 10%
    assert worse["checks"]["3_real_divergences_le_5pct"] is False
    assert mod.gate(worse, tuned)["verdict"] == "PASS"  # on the base, condition 3 is reported, not decisive
    assert mod.gate(worse, tuned)["base_real_divergence_pct"] == 10.0
    assert "fine_tune:3_real_divergences_le_5pct" in mod.gate(base, worse)["failed"]  # decisive on the fine-tune


def test_hf_stopping_where_vllm_goes_on_is_ranked_with_the_eos() -> None:
    hf, vl = _pair(10, differ=1)
    hf[0]["token_ids"] = [1, 7]  # HF ends after one token; vLLM goes on (its token 9 vs EOS 7 at position 1)
    vl[0]["top_logprobs"][1] = [[9, -0.70], [7, -0.75]]
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["hf_choice_rank_in_vllm"] == "2"
    assert report["diverged"][0]["kind"] == "near_tie"


def test_vllm_stopping_where_hf_goes_on_is_ranked_at_vllm_eos() -> None:
    # Symmetric case: vLLM's EOS (token 7, finish "stop") at position 1 has its own logprobs; HF's token 2 is rank 2.
    hf, vl = _pair(10, differ=1)
    vl[0]["token_ids"] = [1, 7]
    vl[0]["top_logprobs"] = [[[1, -0.01], [5, -5.0]], [[7, -0.70], [2, -0.75]]]
    report = _mod().compare(hf, vl)
    assert report["diverged"][0]["first_diff_token"] == 1
    assert report["diverged"][0]["hf_choice_rank_in_vllm"] == "2"
    assert report["diverged"][0]["kind"] == "near_tie"


def test_no_predictions_is_an_error() -> None:
    with pytest.raises(SystemExit, match="no HF predictions"):
        _mod().compare({}, {})
