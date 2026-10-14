"""Distillation fidelity (ADR-022 #6): how closely a student answers the teacher's held-out questions.

The teacher (data.caption_vlm task=qa) wrote one question + answer per image; data.split kept a validation part the
student never trained on. Each student (base / trained on the original labels / trained on the teacher's QA) answers
those questions greedily on vLLM with the same pixels and the same pinned engine, and is scored against the teacher:

  uv run python scripts/qa/distill_fidelity.py run --model <dir> --dataset <val dataset.json> --out <student>.json
  uv run python scripts/qa/distill_fidelity.py score <student>.json [--out report.json]
  uv run python scripts/qa/distill_fidelity.py compare <A>.json <B>.json [--out report.json]   # delta = A - B

Fidelity means agreement with the TEACHER, not correctness: a student that copies a wrong teacher answer scores.
Scores (normalized like VQA / SQuAD: lowercase, no punctuation or articles, collapsed spaces): exact match % and
token F1 % over ALL held-out questions; a request that failed scores 0 (and is counted), so failing on hard images
never raises a score. truncated % = answers cut at MAX_TOKENS (the 0.8B base loops on vLLM after a near tie, a known
trait, reported rather than hidden in F1). The engine is vLLM 0.31 for every student (same KV pin), so the numbers
compare across students and never with an HF-engine number (Qwen3.5-0.8B did not pass the vLLM<->HF gate).

Decision rule, fixed BEFORE any run (bao-tester review 2026-10-14): the headline number is token F1, compared paired
(same questions) as delta = student on the teacher's QA - student on the original labels, 95% CI by paired bootstrap
(1000 resamples, seed 0, as the parity gate). "Distillation helps (fidelity)" only if that CI excludes 0. Exact match
and the delta against the base student are secondary numbers.
Both students must use the same image split (same data.split seed / val_fraction): membership is a hash of the image
name, so the original-labels student never trained on an image of the teacher's validation part either.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from vlm_pipeline.common.paths import project_root

PIXELS = (200704, 1003520)  # the training templates' pixels (same as the parity gate)
MAX_TOKENS = 256  # the teacher's max_new_tokens in the distill workflow
KV_CACHE_GB = 8.0  # pinned for every student, as the parity gate
RESAMPLES, SEED = 1000, 0
CONTAINER_ROOT = "/workspace"
VAL_PARTS = frozenset({"val", "validation"})
ENGINE_KEYS = ("vllm", "dtype", "quantization", "max_model_len", "kv_cache_gb", "enforce_eager", "seed", "chunk",
               "mm_processor_kwargs", "chat_template_kwargs", "container_digest")  # fmt: skip
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def _per_question(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """(exact match, token F1) per question; a failed request (no text) is 0 on both."""
    em, f1 = [], []
    for r in rows:
        text = r.get("student")
        if not isinstance(text, str):
            em.append(0.0)
            f1.append(0.0)
            continue
        em.append(float(normalize(text) == normalize(r["answer"])))
        f1.append(token_f1(text, r["answer"]))
    return em, f1


def _ci(values: list[float]) -> list[float]:
    """Percentile bootstrap 95% CI of the mean, in %."""
    rng = random.Random(SEED)  # noqa: S311 - a reproducible bootstrap, not security
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(RESAMPLES))
    return [round(100 * means[int(0.025 * RESAMPLES)], 2), round(100 * means[int(0.975 * RESAMPLES) - 1], 2)]


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows with ``answer`` (teacher), ``student`` (text or None) and optional ``finish_reason``."""
    if not rows:
        return {"questions": 0}
    em, f1 = _per_question(rows)
    truncated = sum(1 for r in rows if r.get("finish_reason") == "length")
    return {
        "questions": len(rows),
        "failed_requests": sum(1 for r in rows if not isinstance(r.get("student"), str)),
        "truncated_pct": round(100 * truncated / len(rows), 2),
        "token_f1_pct": round(100 * sum(f1) / len(f1), 2),
        "token_f1_ci95": _ci(f1),
        "exact_match_pct": round(100 * sum(em) / len(em), 2),
        "exact_match_ci95": _ci(em),
    }


def _engine(data: dict[str, Any]) -> dict[str, Any]:
    engine = data.get("engine") or {}
    return {k: engine.get(k) for k in ENGINE_KEYS}


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired delta A - B (token F1 headline, exact match secondary) on the same questions and the same engine."""
    key = {str(r["id"]): (r["question"], r["image"], r["answer"]) for r in a["rows"]}
    other = {str(r["id"]): (r["question"], r["image"], r["answer"]) for r in b["rows"]}
    if key != other:
        raise SystemExit("the two runs did not answer the same held-out questions (id, question, image, answer)")
    if (ea := _engine(a)) != (eb := _engine(b)):
        diff = sorted(k for k in ENGINE_KEYS if ea.get(k) != eb.get(k))
        raise SystemExit(f"the two runs used different engine settings: {diff}")
    ids = sorted(key)
    by_a = {str(r["id"]): r for r in a["rows"]}
    by_b = {str(r["id"]): r for r in b["rows"]}
    em_a, f1_a = _per_question([by_a[i] for i in ids])
    em_b, f1_b = _per_question([by_b[i] for i in ids])
    d_f1 = [x - y for x, y in zip(f1_a, f1_b, strict=True)]
    d_em = [x - y for x, y in zip(em_a, em_b, strict=True)]
    ci = _ci(d_f1)
    return {
        "a": a.get("model"), "b": b.get("model"), "questions": len(ids),
        "delta_token_f1_pct": round(100 * sum(d_f1) / len(d_f1), 2), "delta_token_f1_ci95": ci,
        "delta_exact_match_pct": round(100 * sum(d_em) / len(d_em), 2), "delta_exact_match_ci95": _ci(d_em),
        "a_better_on_token_f1": ci[0] > 0, "b_better_on_token_f1": ci[1] < 0,
    }  # fmt: skip


def _host(root: Path, path: str) -> Path:
    rel = Path(path).relative_to(CONTAINER_ROOT) if path.startswith(CONTAINER_ROOT + "/") else Path(path)
    return (root / rel).resolve()


def held_out(root: Path, dataset_json: Path) -> list[dict[str, str]]:
    """The validation part of the teacher's dataset: id, image (host path inside its folder), question, answer."""
    from vlm_pipeline.common.dataset_ref import read_dataset_ref
    from vlm_pipeline.common.local_records import image_path, read_records, select

    ref = read_dataset_ref(dataset_json)
    if not ref.records:
        raise SystemExit("the dataset is not a local records dataset (data.caption_vlm output after data.split)")
    if ref.split not in VAL_PARTS:
        raise SystemExit(f"{dataset_json} is the {ref.split!r} part: fidelity is measured on the held-out val part")
    records = select(read_records(_host(root, ref.records)), part="validation", seed=ref.seed or 0,
                     val_fraction=ref.val_fraction or 0.0, limit=ref.limit)  # fmt: skip
    folder = _host(root, ref.path)
    out = []
    for n, r in enumerate(records):
        if r.generated_by is None or not r.prompt or not isinstance(r.target, str):
            continue  # only the teacher's QA records
        out.append({"id": str(n), "image": str(image_path(folder, r)), "question": r.prompt, "answer": r.target})
    return out


def run(args: argparse.Namespace) -> int:
    from vlm_pipeline.serve.vllm_runner import run_vllm_batch

    root = project_root()
    questions = held_out(root, Path(args.dataset))
    if not questions:
        raise SystemExit("no teacher QA record in the held-out part")
    items = [{"id": q["id"], "image": q["image"], "prompt": q["question"], "max_tokens": MAX_TOKENS} for q in questions]
    res = run_vllm_batch(root, Path(args.model), items, who="bao-fixcode", tag="distill-fidelity",
                         min_pixels=PIXELS[0], max_pixels=PIXELS[1], enable_thinking=False,
                         kv_cache_gb=args.kv_cache_gb)  # fmt: skip
    if not res.ok:
        print(f"batch failed: {res.status} {res.error_code} {res.run_dir}", file=sys.stderr)
        return 1
    answers = {str(r["id"]): r for r in res.rows()}
    rows = [{**q, "student": answers.get(q["id"], {}).get("text"),
             "finish_reason": answers.get(q["id"], {}).get("finish_reason")} for q in questions]  # fmt: skip
    out = {"model": args.model, "dataset": args.dataset, "run_dir": str(res.run_dir.relative_to(root)),
           "engine": res.engine(), "rows": rows}  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_report(out), indent=1))
    return 0


def _report(data: dict[str, Any]) -> dict[str, Any]:
    return {"model": data.get("model"), "engine": _engine(data), **score(data["rows"])}


def _emit(report: dict[str, Any], out: str | None) -> None:
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="distill_fidelity.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="answer the held-out teacher questions with a student (vLLM, GPU lock)")
    r.add_argument("--model", required=True, help="student model dir (merged checkpoint or catalog snapshot)")
    r.add_argument("--dataset", required=True, help="dataset.json of the held-out (val) part")
    r.add_argument("--out", required=True)
    r.add_argument("--kv-cache-gb", type=float, default=KV_CACHE_GB)
    s = sub.add_parser("score", help="score a run file against the teacher's answers")
    s.add_argument("run_file")
    s.add_argument("--out")
    c = sub.add_parser("compare", help="paired delta A - B on the same questions and engine")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)

    def load(path: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return data

    _emit(_report(load(a.run_file)) if a.cmd == "score" else compare(load(a.a), load(a.b)), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
