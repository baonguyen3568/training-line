"""User regexes that cannot hang a job (P020 §3 format_regex, security §B2).

Stdlib ``re`` has no timeout, so the pattern itself is restricted at validation: short, compilable, and none of the
constructs behind catastrophic backtracking or hidden state (nested unbounded quantifiers, alternation under an
unbounded quantifier, backreferences, lookaround, more than 1-2 long repeats). The worker also caps the text length.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

MAX_PATTERN_CHARS = 200
MAX_TEXT_CHARS = 2000  # the answer a pattern is matched against is cut to this (worker)
# Safety rests on all three: these static rules, MAX_TEXT_CHARS in the worker, and the timing test over hostile
# text. Raising MAX_TEXT_CHARS means measuring again (security 2026-10-14).

# The pattern parser is private API, stable since 3.11 (sre_parse before); typed as Any on purpose.
_p: Any = importlib.import_module("re._parser")
_c: Any = importlib.import_module("re._constants")

_REPEATS = {_c.MAX_REPEAT, _c.MIN_REPEAT, getattr(_c, "POSSESSIVE_REPEAT", _c.MAX_REPEAT)}
_LOOKAROUND = {_c.ASSERT, _c.ASSERT_NOT}
_BACKREF = {_c.GROUPREF, getattr(_c, "GROUPREF_EXISTS", _c.GROUPREF)}


class UnsafeRegexError(ValueError):
    """Why a pattern is refused (``reason``).

    too_long | invalid | nested_quantifier | alternation_in_repeat | backreference | lookaround | too_many_repeats.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _children(av: Any) -> list[Any]:
    """Sub-patterns inside one parsed item's argument (groups, branches, repeats)."""
    if isinstance(av, _p.SubPattern):
        return [av]
    if isinstance(av, tuple | list):
        return [c for a in av for c in _children(a)]
    return []


def _walk(pattern: Any, inside_repeat: bool) -> None:
    for op, av in pattern:
        if op in _LOOKAROUND:
            raise UnsafeRegexError("lookaround")
        if op in _BACKREF:
            raise UnsafeRegexError("backreference")
        repeat = op in _REPEATS and av[1] > 1  # x? (at most once) cannot backtrack catastrophically
        if repeat and inside_repeat:
            raise UnsafeRegexError("nested_quantifier")
        if op == _c.BRANCH and inside_repeat:  # (a|a)* backtracks exponentially without any nested quantifier
            raise UnsafeRegexError("alternation_in_repeat")
        for child in _children(av):
            _walk(child, inside_repeat or repeat)


LONG_REPEAT = 10  # a quantifier allowing more than this many repetitions counts against the budget below
MAX_LONG_REPEATS = {True: 2, False: 1}  # anchored at the start: O(n^2) at worst; unanchored: the search loop is one


def _long_repeats(pattern: Any) -> int:
    count = 0
    for op, av in pattern:
        if op in _REPEATS and av[1] > LONG_REPEAT:
            count += 1
        count += sum(_long_repeats(child) for child in _children(av))
    return count


def _anchored(parsed: Any) -> bool:
    first = list(parsed)[:1]
    return bool(first) and first[0][0] == _c.AT and first[0][1] in (_c.AT_BEGINNING, _c.AT_BEGINNING_STRING)


def check_safe_regex(pattern: str) -> str:
    """``pattern`` when it is short, compiles and has no backtracking trap; else UnsafeRegexError."""
    if len(pattern) > MAX_PATTERN_CHARS:
        raise UnsafeRegexError("too_long")
    try:
        parsed = _p.parse(pattern)
        re.compile(pattern)
    except re.error:
        raise UnsafeRegexError("invalid") from None
    _walk(parsed, inside_repeat=False)
    # Polynomial blow-up (security, measured on 2000 chars: ".*.*.*x" unanchored ran for minutes): re cannot be
    # interrupted from Python, so the number of long repeats is capped statically.
    if _long_repeats(parsed) > MAX_LONG_REPEATS[_anchored(parsed)]:
        raise UnsafeRegexError("too_many_repeats")
    return pattern


__all__ = ["MAX_PATTERN_CHARS", "MAX_TEXT_CHARS", "UnsafeRegexError", "check_safe_regex"]
