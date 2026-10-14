"""User regexes (GRPO format_regex) are restricted so they can never backtrack catastrophically (no timeout in re)."""

from __future__ import annotations

import pytest

from vlm_pipeline.common.safe_regex import MAX_PATTERN_CHARS, UnsafeRegexError, check_safe_regex

ALLOWED = [
    r"^\d+(\.\d+)?$",
    r"^\{.*\}$",
    r"^(yes|no)$",
    r"^[A-Z]{2,3}-\d{4}$",
    r"^<answer>[^<]{1,500}</answer>$",
    r"^<think>.*</think>.*$",
    r"^.*.*x",
    r"\bA\b",
]


@pytest.mark.parametrize(
    "pattern",
    [
        r"^\d+(\.\d+)?$",  # a number
        r"^\{.*\}$",  # a JSON object
        r"^(yes|no)$",
        r"^[A-Z]{2,3}-\d{4}$",
        r"^<answer>[^<]{1,500}</answer>$",
        r"^<think>.*</think>.*$",  # 2 long repeats, anchored
        r"^.*.*x",  # 2, anchored: O(n^2)
        r"\bA\b",
    ],
)
def test_ordinary_answer_shapes_are_allowed(pattern: str) -> None:
    assert check_safe_regex(pattern) == pattern


@pytest.mark.parametrize(
    ("pattern", "reason"),
    [
        (r"^(a+)+$", "nested_quantifier"),  # the textbook ReDoS
        (r"^(\w*\s?)*$", "nested_quantifier"),
        (r"^((ab)*c)+$", "nested_quantifier"),
        (r"^(?:x|(y+))*$", "alternation_in_repeat"),
        (r"^(a|a)*$", "alternation_in_repeat"),  # exponential without any nested quantifier
        (r"^(a|ab)+$", "alternation_in_repeat"),
        (r"^(a)\1$", "backreference"),
        (r"^(?P<n>a)(?P=n)$", "backreference"),
        (r"^(?=a)a$", "lookaround"),
        (r"^(?<!b)a$", "lookaround"),
        (r"^(a$", "invalid"),
        (r".*.*x", "too_many_repeats"),  # unanchored: the search loop already repeats once
        (r"^.*.*.*x", "too_many_repeats"),  # measured 1.7 s on 2000 chars
        (r".*.*.*x", "too_many_repeats"),  # measured minutes
        (r"^\w{1,50}\w{1,50}\w{1,50}$", "too_many_repeats"),  # bounded but long repeats count too
        ("a" * (MAX_PATTERN_CHARS + 1), "too_long"),
    ],
)
def test_backtracking_traps_are_refused(pattern: str, reason: str) -> None:
    with pytest.raises(UnsafeRegexError) as err:
        check_safe_regex(pattern)
    assert err.value.reason == reason


def test_every_allowed_pattern_is_fast_on_hostile_text() -> None:
    """Security: whatever passes the check stays fast on the worker's longest text (MAX_TEXT_CHARS)."""
    import re
    import time

    from vlm_pipeline.common.safe_regex import MAX_TEXT_CHARS

    hostile = [c * MAX_TEXT_CHARS + "!" for c in ("a", "1", "<", " ")] + ["<think>" * (MAX_TEXT_CHARS // 7)]
    for pattern in ALLOWED:
        check_safe_regex(pattern)
        compiled = re.compile(pattern)
        for text in hostile:
            start = time.perf_counter()
            compiled.search(text)
            assert time.perf_counter() - start < 0.05, (pattern, text[:10])
