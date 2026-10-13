"""CORD-v2 receipt-parsing metrics, ported from Donut (clovaai/donut) so scores are comparable with its paper.

Reference (read 2026-10-12): ``donut/model.py::DonutModel.token2json`` and
``donut/util.py::JSONParseEvaluator`` (``normalize_dict``, ``flatten``, ``cal_f1``, ``cal_acc``).
Pure Python, host-safe (no torch, no third-party deps). Deviations from the reference are listed in each docstring.

Metrics (P001a §7.2, ADR-002 registry):
- ``field_f1``: micro F1 over leaf ``(path, value)`` pairs, summed over the whole set (primary, higher is better).
- ``json_exact``: % of samples whose normalised JSON equals the ground truth.
- ``nted_accuracy``: mean of ``max(0, 1 - nTED)`` per sample (Donut's "accuracy").
- ``parse_fail_rate``: % of outputs that ``token2json`` cannot turn into a dict (lower is better).

Chat models (Qwen, Gemma… trained to answer the ``gt_parse`` JSON itself) are scored by the SAME metrics on the dict
``json_answer`` reads from their text (``score_json_texts``), so their numbers compare with Nemotron-Parse's.

Scale: the metric functions return 0-1 (like ``scoring.anls``); ``CordScores`` is the report record, in percent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

TEXT_SEQUENCE_KEY = "text_sequence"
SEP = "<sep/>"
LEAF = "<leaf>"

Json = dict[str, Any]


# ----------------------------------------------------------------------------------------- token2json
def token2json(tokens: str, *, categorical: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Parse Donut-style ``<s_key>value</s_key>`` tokens back into JSON.

    Returns ``{"text_sequence": <rest>}`` when nothing could be parsed (= parse failure).
    ``categorical`` holds special tokens like ``"<menu/>"`` that are unwrapped to ``"menu"``.
    Deviation: the closing tag is located with the same case-insensitive match used to find it (the reference
    slices with a case-sensitive ``str.find``, which mis-cuts on a case mismatch) and the key is regex-escaped.
    """
    items = _parse_seq(tokens, categorical)
    if not items:
        return {TEXT_SEQUENCE_KEY: _strip_unclosed(tokens)}
    return items[0]


def _strip_unclosed(tokens: str) -> str:
    """What the reference leaves in ``text_sequence``: the input minus start tags that were never closed."""
    while (start := re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)) is not None:
        if re.search(rf"</s_{re.escape(start.group(1))}>", tokens, re.IGNORECASE) is not None:
            break
        tokens = tokens.replace(start.group(), "")
    return tokens


def _parse_seq(tokens: str, categorical: frozenset[str]) -> list[dict[str, Any]]:
    """Reference ``token2json(..., is_inner_value=True)``: a ``<sep/>``-separated sequence of dicts."""
    output: dict[str, Any] = {}
    while tokens:
        start = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start is None:
            break
        key = start.group(1)
        end = re.search(rf"</s_{re.escape(key)}>", tokens, re.IGNORECASE)
        if end is None:
            tokens = tokens.replace(start.group(), "")
            continue
        content_match = re.search(
            f"{re.escape(start.group())}(.*?){re.escape(end.group())}", tokens, re.IGNORECASE | re.DOTALL
        )
        if content_match is not None:
            content = content_match.group(1).strip()
            if "<s_" in content and "</s_" in content:
                value = _parse_seq(content, categorical)
                if value:
                    output[key] = value[0] if len(value) == 1 else value
            else:
                leaves = []
                for raw in content.split(SEP):
                    leaf = raw.strip()
                    if leaf in categorical and leaf.startswith("<") and leaf.endswith("/>"):
                        leaf = leaf[1:-2]
                    leaves.append(leaf)
                output[key] = leaves[0] if len(leaves) == 1 else leaves
            tokens = tokens[content_match.end() :].strip()
        else:
            tokens = tokens[end.end() :].strip()
        if tokens.startswith(SEP):
            return [output, *_parse_seq(tokens[len(SEP) :], categorical)]
    return [output] if output else []


def is_parse_failure(parsed: dict[str, Any]) -> bool:
    """True when ``token2json`` produced no structured field (only the raw ``text_sequence`` fallback)."""
    return not parsed or set(parsed) == {TEXT_SEQUENCE_KEY}


# --------------------------------------------------------------------------------------- normalisation
def normalize(data: Any) -> Any:
    """Donut ``normalize_dict``.

    Keys sorted by (len, name), values wrapped in lists, empty values dropped, scalars become stripped strings;
    lists of non-dicts keep only str/int/float items (bool excluded, as in Donut).
    """
    if not data:
        return {}
    if isinstance(data, dict):
        out: Json = {}
        for key in sorted(data, key=lambda k: (len(k), k)):
            value = normalize(data[key])
            if value:
                out[key] = value if isinstance(value, list) else [value]
        return out
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [n for n in (normalize(item) for item in data) if n]
        return [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    return [str(data).strip()]


def flatten(data: Any) -> list[tuple[str, str]]:
    """Donut ``flatten``: dotted paths, list items share their parent path, leaves become ``(path, value)``."""
    pairs: list[tuple[str, str]] = []

    def walk(value: Any, key: str) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, f"{key}.{child_key}" if key else child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        else:
            pairs.append((key, value))

    walk(data, "")
    return pairs


# ---------------------------------------------------------------------------------------------- F1
def field_f1(preds: Sequence[Any], answers: Sequence[Any]) -> float:
    """Micro F1 over all samples (Donut ``cal_f1``), on the 0-1 scale. Duplicates match one-to-one.

    Deviation: the reference divides by zero when neither side has any field; we return 1.0 in that case.
    """
    if len(preds) != len(answers):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(answers)} answers")
    counts = [field_f1_counts(pred, answer) for pred, answer in zip(preds, answers, strict=True)]
    return f1_from_counts(counts)


def field_f1_counts(pred: Any, answer: Any) -> tuple[int, int]:
    """``(true positives, errors)`` of one sample; micro F1 of a set is computed from the summed counts.

    Per-sample counts let a bootstrap resample images without re-matching fields (``f1_from_counts``).
    """
    tp = errors = 0
    remaining = flatten(normalize(answer))
    for pair in flatten(normalize(pred)):
        if pair in remaining:
            tp += 1
            remaining.remove(pair)
        else:
            errors += 1
    return tp, errors + len(remaining)


def f1_from_counts(counts: Sequence[tuple[int, int]]) -> float:
    """Micro F1 (0-1) from per-sample ``field_f1_counts``."""
    tp = sum(c[0] for c in counts)
    errors = sum(c[1] for c in counts)
    if tp + errors == 0:
        return 1.0
    return tp / (tp + errors / 2)


def json_exact(preds: Sequence[Any], answers: Sequence[Any]) -> float:
    """Fraction (0-1) of samples whose normalised JSON equals the normalised ground truth."""
    if len(preds) != len(answers):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(answers)} answers")
    if not preds:
        return 0.0
    hits = sum(1 for p, a in zip(preds, answers, strict=True) if normalize(p) == normalize(a))
    return hits / len(preds)


# ---------------------------------------------------------------------------------------------- nTED
@dataclass
class _Node:
    label: str
    children: list[_Node]


def _tree(data: Any, label: str = "<root>") -> _Node:
    """Donut ``construct_tree_from_dict`` on already-normalised data."""
    node = _Node(label, [])
    if isinstance(data, dict):
        node.children = [_tree(value, key) for key, value in data.items()]
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            node.children = [_tree(item, "<subtree>") for item in data]
        else:
            node.children = [_Node(f"{LEAF}{item}", []) for item in data]
    else:
        raise TypeError(f"unexpected value under {label!r}: {data!r}")
    return node


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _insert_remove_cost(node: _Node) -> int:
    return len(node.label[len(LEAF) :]) if node.label.startswith(LEAF) else 1


def _update_cost(a: _Node, b: _Node) -> int:
    a_leaf, b_leaf = a.label.startswith(LEAF), b.label.startswith(LEAF)
    if a_leaf and b_leaf:
        return _levenshtein(a.label[len(LEAF) :], b.label[len(LEAF) :])
    if a_leaf:
        return 1 + len(a.label[len(LEAF) :])
    if b_leaf:
        return 1 + len(b.label[len(LEAF) :])
    return int(a.label != b.label)


def _postorder(root: _Node) -> tuple[list[_Node], list[int], list[int]]:
    """Nodes in post-order, leftmost-leaf index of each, and the keyroots (Zhang-Shasha)."""
    nodes: list[_Node] = []
    lml: list[int] = []

    def walk(node: _Node) -> int:
        first = -1
        for child in node.children:
            leftmost = walk(child)
            if first < 0:
                first = leftmost
        nodes.append(node)
        index = len(nodes) - 1
        lml.append(first if first >= 0 else index)
        return lml[index]

    walk(root)
    seen: dict[int, int] = {}
    for i, leftmost in enumerate(lml):
        seen[leftmost] = i  # last (highest) node per leftmost leaf is the keyroot
    return nodes, lml, sorted(seen.values())


def tree_edit_distance(a: _Node, b: _Node) -> int:
    """Ordered tree edit distance (Zhang & Shasha 1989) with Donut's cost functions; same result as ``zss``."""
    an, al, akr = _postorder(a)
    bn, bl, bkr = _postorder(b)
    td = [[0] * len(bn) for _ in an]
    for i in akr:
        for j in bkr:
            ioff, joff = al[i] - 1, bl[j] - 1
            m, n = i - ioff, j - joff
            fd = [[0] * (n + 1) for _ in range(m + 1)]
            for x in range(1, m + 1):
                fd[x][0] = fd[x - 1][0] + _insert_remove_cost(an[x + ioff])
            for y in range(1, n + 1):
                fd[0][y] = fd[0][y - 1] + _insert_remove_cost(bn[y + joff])
            for x in range(1, m + 1):
                for y in range(1, n + 1):
                    p, q = x + ioff, y + joff
                    remove = fd[x - 1][y] + _insert_remove_cost(an[p])
                    insert = fd[x][y - 1] + _insert_remove_cost(bn[q])
                    if al[p] == al[i] and bl[q] == bl[j]:
                        fd[x][y] = min(remove, insert, fd[x - 1][y - 1] + _update_cost(an[p], bn[q]))
                        td[p][q] = fd[x][y]
                    else:
                        fd[x][y] = min(remove, insert, fd[al[p] - 1 - ioff][bl[q] - 1 - joff] + td[p][q])
    return td[-1][-1]


def nted_accuracy_one(pred: Any, answer: Any) -> float:
    """Donut ``cal_acc``: ``max(0, 1 - TED(pred, gt) / TED(empty, gt))``.

    Deviation: the reference divides by zero for an empty ground truth; we return 1.0 if the prediction is
    empty too, else 0.0.
    """
    gt = _tree(normalize(answer))
    empty = _tree(normalize({}))
    denom = tree_edit_distance(empty, gt)
    pr = _tree(normalize(pred))
    if denom == 0:
        return 1.0 if tree_edit_distance(pr, gt) == 0 else 0.0
    return max(0.0, 1.0 - tree_edit_distance(pr, gt) / denom)


def nted_accuracy(preds: Sequence[Any], answers: Sequence[Any]) -> float:
    if len(preds) != len(answers):
        raise ValueError(f"length mismatch: {len(preds)} predictions vs {len(answers)} answers")
    if not preds:
        return 0.0
    return sum(nted_accuracy_one(p, a) for p, a in zip(preds, answers, strict=True)) / len(preds)


# ------------------------------------------------------------------------------------------- report
@dataclass(frozen=True)
class CordScores:
    """All values on the 0-100 scale used by benchmark records (``scale`` = percent)."""

    field_f1: float
    json_exact: float
    nted_accuracy: float
    parse_fail_rate: float
    n_samples: int


def score_sequences(
    sequences: Sequence[str], answers: Sequence[Any], *, categorical: frozenset[str] = frozenset()
) -> CordScores:
    """Score raw decoded sequences (task prompt / special tokens already removed) against ``gt_parse`` dicts.

    A parse failure counts as an empty prediction for F1/exact/nTED, so failures are penalised, not skipped.
    """
    if len(sequences) != len(answers):
        raise ValueError(f"length mismatch: {len(sequences)} sequences vs {len(answers)} answers")
    parsed = [token2json(s, categorical=categorical) for s in sequences]
    return _score([None if is_parse_failure(p) else p for p in parsed], answers)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_answer(text: str) -> Json | None:
    """The JSON object a chat model answered, or None (parse failure).

    Reads a fenced block (```json ... ```) if there is one, else the text; text before the first ``{`` (a sentence)
    and after the object is ignored. Only the OUTERMOST object (the first ``{``) counts: a truncated answer must be a
    parse failure, never an inner item that happens to close (research review 2026-10-13). A top-level list or scalar
    is a failure too: the target is an object.
    """
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate, start)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def score_json_texts(texts: Sequence[str], answers: Sequence[Any]) -> CordScores:
    """Score chat-model answers (JSON text) against ``gt_parse`` dicts with the metrics of ``score_sequences``."""
    if len(texts) != len(answers):
        raise ValueError(f"length mismatch: {len(texts)} texts vs {len(answers)} answers")
    return _score([json_answer(t) for t in texts], answers)


ANSWER_FORMATS = ("donut", "json")  # Donut-style <s_key> tokens (Nemotron-Parse) | JSON text (chat models)


def prediction_of(row: dict[str, Any]) -> Json | None:
    """The predicted dict of one prediction record (``sequence`` + ``answer_format``, default donut); None = failure."""
    fmt = row.get("answer_format", "donut")
    if fmt == "json":
        return json_answer(row["sequence"])
    if fmt == "donut":
        parsed = token2json(row["sequence"])
        return None if is_parse_failure(parsed) else parsed
    raise ValueError(f"unknown answer_format {fmt!r} (expected one of {ANSWER_FORMATS})")


def score_rows(rows: Sequence[dict[str, Any]]) -> CordScores:
    """Score prediction records of ONE format (a run never mixes a Parse and a chat model's answers)."""
    formats = {r.get("answer_format", "donut") for r in rows}
    if len(formats) > 1:
        raise ValueError(f"records mix answer formats {sorted(formats)}")
    return _score([prediction_of(r) for r in rows], [r["gt_parse"] for r in rows])


def _score(parsed: Sequence[Json | None], answers: Sequence[Any]) -> CordScores:
    """A parse failure (None) counts as an empty prediction for F1/exact/nTED: penalised, not skipped."""
    preds = [{} if p is None else p for p in parsed]
    n = len(parsed)
    return CordScores(
        field_f1=100 * field_f1(preds, answers),
        json_exact=100 * json_exact(preds, answers),
        nted_accuracy=100 * nted_accuracy(preds, answers),
        parse_fail_rate=100 * (sum(p is None for p in parsed) / n if n else 0.0),
        n_samples=n,
    )
