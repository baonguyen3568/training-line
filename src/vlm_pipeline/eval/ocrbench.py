"""OCRBench scoring, ported from the reference so scores are comparable with the paper and model cards.

Reference (read 2026-10-13): Yuliang-Liu/MultimodalOCR ``OCRBench/example.py`` (scoring loop of ``eval_worker`` and
the per-type totals). Rule: a sample scores 1 when a ground-truth answer, normalised, is a SUBSTRING of the normalised
prediction; HME100k (handwritten math) drops every space instead of lower-casing. Final score = sum over the 1000
samples (0-1000); benchmark records use ``value = score / 10`` (0-100, frontend-metrics-contract).
Deviation option: ``fold_fullwidth=True`` also folds fullwidth ASCII (U+FF01-FF5E) and the ideographic space, as
lmms-eval 0.7 ``tasks/ocrbench/utils.py`` does; the default (False) is the reference. The setting is recorded.
Pure Python, host-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

HME = "HME100k"
# Security conditions for echo840/OCRBench@92a54bd1 (eval_only, 2026-10-13): reports cite the paper; only aggregate
# scores are published (no image or question leaves the machine); images decode under serve.trial.open_image rules.
CITATION = (
    "Liu et al., OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models, "
    "Science China Information Sciences 2024 (arXiv:2305.07895)"
)
# Per-type totals of the 1000-sample set (reference print-out); recognition = the first six.
TYPE_TOTALS = {
    "Regular Text Recognition": 50,
    "Irregular Text Recognition": 50,
    "Artistic Text Recognition": 50,
    "Handwriting Recognition": 50,
    "Digit String Recognition": 50,
    "Non-Semantic Text Recognition": 50,
    "Scene Text-centric VQA": 200,
    "Doc-oriented VQA": 200,
    "Key Information Extraction": 200,
    "Handwritten Mathematical Expression Recognition": 100,
}
RECOGNITION_TYPES = tuple(list(TYPE_TOTALS)[:6])


def _fold_fullwidth(text: str) -> str:
    out = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def _normalise(text: str, *, dataset: str, fold_fullwidth: bool) -> str:
    if dataset == HME:
        return text.strip().replace("\n", " ").replace(" ", "")
    if fold_fullwidth:
        text = _fold_fullwidth(text)
    return text.lower().strip().replace("\n", " ")


def score_one(prediction: str, answers: str | Sequence[str], dataset: str, *, fold_fullwidth: bool = False) -> int:
    """1 if any normalised answer is contained in the normalised prediction, else 0 (reference rule)."""
    candidates = [answers] if isinstance(answers, str) else list(answers)
    predict = _normalise(prediction.strip(), dataset=dataset, fold_fullwidth=fold_fullwidth)
    return int(any(_normalise(a, dataset=dataset, fold_fullwidth=fold_fullwidth) in predict for a in candidates))


@dataclass(frozen=True)
class OcrBenchScores:
    """Only the full 1000-sample set is comparable with model cards (e.g. Qwen3.5 OCRBench); a subset is not."""

    final: int  # 0-1000 on the full set (sum of per-sample scores)
    n_samples: int
    accuracy: float  # 0-100: final / n_samples * 100 (= final / 10 on the full set)
    recognition: int
    by_type: dict[str, int]


def score_samples(rows: Sequence[dict[str, Any]], *, fold_fullwidth: bool = False) -> OcrBenchScores:
    """Rows carry ``prediction``, ``answer`` (str or list), ``dataset`` and ``question_type`` (dataset columns)."""
    by_type = dict.fromkeys(TYPE_TOTALS, 0)
    final = 0
    for row in rows:
        s = score_one(row["prediction"], row["answer"], row["dataset"], fold_fullwidth=fold_fullwidth)
        qtype = row["question_type"]
        if qtype not in by_type:
            raise ValueError(f"unknown OCRBench question_type {qtype!r}")
        by_type[qtype] += s
        final += s
    n = len(rows)
    return OcrBenchScores(
        final=final,
        n_samples=n,
        accuracy=100 * final / n if n else 0.0,
        recognition=sum(by_type[t] for t in RECOGNITION_TYPES),
        by_type=by_type,
    )
