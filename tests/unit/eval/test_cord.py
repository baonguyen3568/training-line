"""Tests for ``vlm_pipeline.eval.cord`` (CORD-v2 metrics, Donut-compatible). Expected values are hand-computed
from the Donut reference semantics (clovaai/donut ``token2json`` / ``JSONParseEvaluator``)."""

from __future__ import annotations

from typing import Any

import pytest

from vlm_pipeline.eval import cord
from vlm_pipeline.eval.cord import (
    TEXT_SEQUENCE_KEY,
    field_f1,
    is_parse_failure,
    json_exact,
    nted_accuracy_one,
    score_sequences,
    token2json,
)


# ------------------------------------------------------------------------------------------ token2json
@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ("<s_menu><s_nm>Latte</s_nm><s_cnt>2</s_cnt></s_menu>", {"menu": {"nm": "Latte", "cnt": "2"}}),
        ("<s_menu><s_nm>A</s_nm><sep/><s_nm>B</s_nm></s_menu>", {"menu": [{"nm": "A"}, {"nm": "B"}]}),
        ("<s_a>x<sep/>y</s_a>", {"a": ["x", "y"]}),
        ("<s_a> padded </s_a>", {"a": "padded"}),
        ("<s_price>10</S_PRICE>", {"price": "10"}),  # close tag is matched case-insensitively
        ("junk <s_a>1</s_a> trailing", {"a": "1"}),  # text outside tags is dropped
    ],
)
def test_token2json(tokens: str, expected: dict[str, Any]) -> None:
    assert token2json(tokens) == expected


def test_categorical_tokens_are_unwrapped() -> None:
    assert token2json("<s_kind><cash/></s_kind>", categorical=frozenset({"<cash/>"})) == {"kind": "cash"}
    assert token2json("<s_kind><cash/></s_kind>") == {"kind": "<cash/>"}  # unknown token stays literal


@pytest.mark.parametrize("tokens", ["", "just text", "<s_a>never closed", "<s_a></s_b>"])
def test_unparseable_sequences_are_failures(tokens: str) -> None:
    parsed = token2json(tokens)
    assert TEXT_SEQUENCE_KEY in parsed
    assert is_parse_failure(parsed)


def test_parsed_dict_is_not_a_failure() -> None:
    assert not is_parse_failure(token2json("<s_a>1</s_a>"))


# ------------------------------------------------------------------------------------------------- F1
def test_f1_perfect_and_half() -> None:
    assert field_f1([{"a": "1", "b": "2"}], [{"a": "1", "b": "2"}]) == pytest.approx(1.0)
    assert field_f1([{"a": "1", "b": "2"}], [{"a": "1", "b": "3"}]) == pytest.approx(0.5)  # tp=1, fp+fn=2


def test_f1_is_micro_over_the_whole_set() -> None:
    preds = [{"a": "1", "b": "2"}, {}]
    answers = [{"a": "1", "b": "2"}, {"a": "x", "b": "y"}]
    assert field_f1(preds, answers) == pytest.approx(2 / (2 + 2 / 2))  # not the mean of per-sample F1 (0.5)


def test_f1_duplicates_match_one_to_one() -> None:
    assert field_f1([{"a": ["x", "x"]}], [{"a": ["x"]}]) == pytest.approx(1 / (1 + 1 / 2))


def test_f1_list_order_does_not_matter_but_paths_do() -> None:
    gt = {"menu": [{"nm": "A"}, {"nm": "B"}]}
    assert field_f1([{"menu": [{"nm": "B"}, {"nm": "A"}]}], [gt]) == pytest.approx(1.0)
    assert field_f1([{"nm": "A"}], [{"menu": {"nm": "A"}}]) == pytest.approx(0.0)


def test_whitespace_is_only_stripped_like_donut() -> None:
    """Donut strips leaf values but does not collapse inner whitespace; we keep that for comparability."""
    assert field_f1([{"a": " x "}], [{"a": "x"}]) == pytest.approx(1.0)
    assert field_f1([{"a": "x  y"}], [{"a": "x y"}]) == pytest.approx(0.0)


def test_empty_values_are_ignored() -> None:
    assert field_f1([{"a": "1", "b": ""}], [{"a": "1"}]) == pytest.approx(1.0)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        field_f1([{}], [])


# ---------------------------------------------------------------------------------------- exact & nTED
def test_json_exact_ignores_key_order_and_padding() -> None:
    assert json_exact([{"b": "2", "a": " 1"}], [{"a": "1", "b": "2"}]) == pytest.approx(1.0)
    assert json_exact([{"a": "1"}], [{"a": "1", "b": "2"}]) == pytest.approx(0.0)


def test_nted_identical_empty_and_one_char() -> None:
    gt = {"a": "abcd"}
    assert nted_accuracy_one(gt, gt) == pytest.approx(1.0)
    assert nted_accuracy_one({}, gt) == pytest.approx(0.0)
    # tree(gt) = root -> a -> <leaf>abcd ; building it costs 1 (node a) + 4 (leaf chars) = 5; one substitution = 1
    assert nted_accuracy_one({"a": "abce"}, gt) == pytest.approx(1 - 1 / 5)


def test_nted_is_clamped_at_zero() -> None:
    assert nted_accuracy_one({"x": "a very long wrong answer"}, {"a": "b"}) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------------- end to end
def test_score_sequences_counts_parse_failures_as_empty_predictions() -> None:
    answers = [{"menu": {"nm": "Latte"}}, {"total": "5"}]
    scores = score_sequences(["<s_menu><s_nm>Latte</s_nm></s_menu>", "garbage output"], answers)
    assert scores.n_samples == 2
    assert scores.parse_fail_rate == pytest.approx(50.0)
    assert scores.field_f1 == pytest.approx(100 * 1 / (1 + 1 / 2))
    assert scores.json_exact == pytest.approx(50.0)
    assert scores.nted_accuracy == pytest.approx(50.0)


def test_tree_edit_distance_matches_the_zss_golden_set() -> None:
    """Golden distances from the independent ``zss`` library (scripts/qa/make_ted_golden.py)."""
    import json
    from pathlib import Path

    from vlm_pipeline.eval.cord import _tree, tree_edit_distance

    golden = json.loads((Path(__file__).parents[2] / "fixtures" / "eval" / "ted_golden.json").read_text())
    assert len(golden["pairs"]) >= 30
    for pair in golden["pairs"]:
        assert tree_edit_distance(_tree(pair["a"]), _tree(pair["b"])) == pair["distance"], pair


# --------------------------------------------------------------------- chat models (JSON answers, R2 / luồng C)
GT = {"menu": {"nm": "TEA", "price": "10.000"}, "total": {"total_price": "10.000"}}


@pytest.mark.parametrize(
    "text",
    [
        '{"menu": {"nm": "TEA", "price": "10.000"}, "total": {"total_price": "10.000"}}',
        '```json\n{"menu": {"nm": "TEA", "price": "10.000"}, "total": {"total_price": "10.000"}}\n```',
        'Here are the fields:\n{"menu": {"nm": "TEA", "price": "10.000"}, "total": {"total_price": "10.000"}} Done.',
    ],
)
def test_json_answer_reads_the_object_around_chatter_and_fences(text: str) -> None:
    assert cord.json_answer(text) == GT
    assert cord.score_json_texts([text], [GT]).field_f1 == 100


TRUNCATED = '{"menu": [{"nm": "TEA", "price": "10.000"}, {"nm": "CO'  # an inner item closes, the answer does not


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]", '"just a string"', '{"menu": {"nm": "TEA"', TRUNCATED])
def test_non_object_or_broken_json_is_a_parse_failure(text: str) -> None:
    assert cord.json_answer(text) is None
    scores = cord.score_json_texts([text], [GT])
    assert scores.parse_fail_rate == 100
    assert scores.field_f1 == 0  # counted as an empty prediction, never skipped


def test_json_and_token_scoring_agree_on_the_same_prediction() -> None:
    """A chat model and Parse that predict the same dict get the same numbers (R2 compares with R0 directly)."""
    import json

    pred = {"menu": {"nm": "TEA", "price": "9.000"}, "total": {"total_price": "10.000"}}
    tokens = (
        "<s_menu><s_nm>TEA</s_nm><s_price>9.000</s_price></s_menu>"
        "<s_total><s_total_price>10.000</s_total_price></s_total>"
    )
    assert cord.score_json_texts([json.dumps(pred)], [GT]) == cord.score_sequences([tokens], [GT])


def test_score_rows_reads_each_format_and_refuses_a_mix() -> None:
    import json

    chat = {"answer_format": "json", "sequence": json.dumps(GT), "gt_parse": GT}
    parse = {"sequence": "<s_total><s_total_price>10.000</s_total_price></s_total>", "gt_parse": GT}  # donut default
    assert cord.score_rows([chat]).field_f1 == 100
    assert 0 < cord.score_rows([parse]).field_f1 < 100
    with pytest.raises(ValueError, match="mix answer formats"):
        cord.score_rows([chat, parse])
    with pytest.raises(ValueError, match="unknown answer_format"):
        cord.prediction_of({"answer_format": "yaml", "sequence": "", "gt_parse": GT})
