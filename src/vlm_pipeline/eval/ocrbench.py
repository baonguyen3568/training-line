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

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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


SUBSET_PATH = Path(__file__).resolve().parents[3] / "configs" / "eval" / "subsets" / "regression_500" / "ocrbench.json"


def load_subset(questions: Sequence[str], path: Path = SUBSET_PATH) -> list[int]:
    """Row indices of the committed regression subset (P002 B5: 500 questions, stratified by question_type, seeded).

    Level "quick". Each row's question must hash to the stored ``q_sha256`` (first 16 hex of sha256): a reordered or
    different dataset revision fails instead of silently scoring other questions. Not comparable with model cards.
    """
    spec = json.loads(path.read_text(encoding="utf-8"))
    rows = [int(s["row"]) for s in spec["samples"]]
    for sample in spec["samples"]:
        row = int(sample["row"])
        if row >= len(questions):
            raise ValueError(f"subset row {row} beyond the dataset ({len(questions)} rows)")
        if hashlib.sha256(questions[row].encode()).hexdigest()[:16] != sample["q_sha256"]:
            raise ValueError(f"subset row {row}: question differs from the pinned revision (reordered dataset?)")
    return rows


def paired_accuracy(
    candidate: Sequence[int], reference: Sequence[int], *, resamples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Accuracy(candidate) - accuracy(reference) in points (0-100) on the same samples, 95% bootstrap CI over samples.

    For the quant gate (FP8 vs its bf16 parent, P002 B3): per-sample 0/1 scores of both runs, in the same order.
    """
    import random

    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired comparison needs two non-empty score lists of the same length")
    n = len(candidate)
    rng = random.Random(seed)  # noqa: S311 - statistics, seeded for reproducibility
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100 * (sum(candidate[i] for i in idx) - sum(reference[i] for i in idx)) / n)
    diffs.sort()
    return {
        "delta": round(100 * (sum(candidate) - sum(reference)) / n, 2),
        "ci95": [round(diffs[int(0.025 * resamples)], 2), round(diffs[int(0.975 * resamples) - 1], 2)],
        "n": n,
        "resamples": resamples,
        "seed": seed,
    }


# ------------------------------------------------------------------ extracted copy + vLLM batch (R3 gate, node)
MANIFEST = "MANIFEST.sha256"
MAX_TOKENS = 256  # answers are short; truncation is reported
KV_CACHE_GB = 8.0  # pinned, the same for both runs of a pair (KV size changes vLLM scheduling, hence numerics)
# Engine settings two runs must share to be paired (only the quantization may differ).
SAME_ENGINE = (
    "vllm", "dtype", "max_model_len", "enforce_eager", "seed", "sampling", "max_num_seqs",
    "max_num_batched_tokens", "enable_chunked_prefill", "chunk", "kv_cache_gb", "chat_template_kwargs",
    "mm_processor_kwargs", "container_digest",
)  # fmt: skip


class OcrBenchDataError(ValueError):
    """The extracted copy is missing or differs from its manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (images/\d+\.(?:jpg|png|webp)|questions\.jsonl)")
_MANIFEST_MAX_BYTES = 4 * 1024 * 1024


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def verify_manifest(data: Path, pinned_sha256: str | None = None) -> str:
    """Check the extracted copy is THE known one; returns its manifest's sha256.

    The manifest (a small regular file, hashing to the catalog's pin when given) lists only allowed names, each a
    regular file with that sha256.

    Any problem (missing file, link, malformed line, ``..``, other hash) is OcrBenchDataError, never a crash.
    """
    manifest = data / MANIFEST
    if not _regular(manifest) or manifest.stat().st_size > _MANIFEST_MAX_BYTES:
        raise OcrBenchDataError(f"{data}: no {MANIFEST} (not extracted)")
    own = _sha256_file(manifest)
    if pinned_sha256 is not None and own != pinned_sha256:
        raise OcrBenchDataError(f"{data}: {MANIFEST} is not the catalog's")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise OcrBenchDataError(f"{data}: unreadable {MANIFEST}") from exc
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise OcrBenchDataError(f"{data}: malformed {MANIFEST} line")
        target = data / match[2]
        if not _regular(target) or _sha256_file(target) != match[1]:
            raise OcrBenchDataError(f"{target}: differs from {MANIFEST}")
    return own


def questions(data: Path, *, subset: bool = True) -> list[dict[str, Any]]:
    """Rows of questions.jsonl: the committed regression_500 subset (q_sha256 checked), or all 1000."""
    rows = [json.loads(line) for line in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    if not subset:
        return rows
    return [rows[i] for i in load_subset([r["question"] for r in rows])]


def batch_requests(rows: Sequence[dict[str, Any]], data: Path, max_tokens: int = MAX_TOKENS) -> list[dict[str, Any]]:
    """Batch items for the vLLM runner: the question verbatim (VLMEvalKit protocol), one image."""
    return [
        {"id": str(r["row"]), "image": str(data / r["image"]), "prompt": r["question"], "max_tokens": max_tokens}
        for r in rows
    ]


def score_batch(rows: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-question 0/1 scores of a batch's outputs (matched by id; an image error scores 0 and is kept)."""
    by_id = {str(o["id"]): o for o in outputs}
    scored = []
    for r in rows:
        out = by_id.get(str(r["row"]), {"error": "missing"})  # an answer the batch did not return scores 0
        scored.append(
            {
                "row": r["row"],
                "question_type": r["question_type"],
                "dataset": r["dataset"],
                "score": 0 if "error" in out else score_one(out.get("text", ""), r["answers"], r["dataset"]),
                "error": out.get("error"),
                "finish_reason": out.get("finish_reason"),
                "gen_tokens": out.get("gen_tokens"),
            }
        )
    return scored


def summary(scored: Sequence[dict[str, Any]]) -> dict[str, float]:
    n = len(scored)
    return {
        "accuracy": round(100 * sum(s["score"] for s in scored) / n, 2),
        "truncated_rate": round(100 * sum(s["finish_reason"] == "length" for s in scored) / n, 2),
        "error_rate": round(100 * sum(s["error"] is not None for s in scored) / n, 2),
        "gen_tokens_mean": round(sum(s["gen_tokens"] or 0 for s in scored) / n, 1),
    }


def engine_mismatch(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Engine settings (``SAME_ENGINE``) on which two runs differ: such runs are never paired."""
    return [f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in SAME_ENGINE if a.get(k) != b.get(k)]
