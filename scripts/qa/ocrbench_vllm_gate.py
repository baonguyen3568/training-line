"""OCRBench regression_500 on vLLM (offline batch, B7-3) and the FP8-vs-bf16 gate of R3 (P002 B3, gates.yaml).

  # one model -> per-question scores + engine (same engine settings for both models)
  uv run python scripts/qa/ocrbench_vllm_gate.py run --model <dir> --label bf16-a [--quantization modelopt] \
      --out outputs/tester/r3/bf16-a.json
  # paired comparison + verdict (candidate vs reference), or a noise check of two runs of the same model
  uv run python scripts/qa/ocrbench_vllm_gate.py compare outputs/tester/r3/bf16-a.json outputs/tester/r3/fp8.json \
      --precision fp8 [--noise outputs/tester/r3/bf16-b.json]

Scoring: eval.ocrbench (reference substring rule); questions = the committed regression_500 subset, verified by
q_sha256 against the extracted copy, whose MANIFEST.sha256 is checked first and recorded (security, 2026-10-13).
Published numbers are aggregates only, with eval.ocrbench.CITATION; a subset score is not comparable with model cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from vlm_pipeline.common.paths import project_root
from vlm_pipeline.eval.gate import BenchmarkResult, quant_gate
from vlm_pipeline.eval.ocrbench import CITATION, load_subset, paired_accuracy, score_one

DATA = Path("data") / "eval" / "ocrbench-92a54bd1"
MAX_TOKENS = 256  # OCRBench answers are short; truncation is reported, not hidden
KV_CACHE_GB = 8.0  # pinned, the SAME for both runs of a pair (KV size changes scheduling, hence numerics)
PIXELS = (200704, 1003520)  # Qwen3.5 training templates (f95897c): the same image budget as training
# Engine settings that must be equal for a paired comparison (quantization is what differs).
SAME_ENGINE = (
    "vllm",
    "dtype",
    "max_model_len",
    "enforce_eager",
    "seed",
    "sampling",  # greedy (temperature 0, n 1) on both sides
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "chunk",
    "kv_cache_gb",
    "chat_template_kwargs",
    "mm_processor_kwargs",
    "container_digest",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(data: Path) -> str:
    """Every file listed in MANIFEST.sha256 matches; returns the manifest's own sha256 (recorded with the scores)."""
    manifest = data / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if _sha256(data / rel) != digest:
            raise SystemExit(f"{data / rel}: differs from {manifest}; re-extract (scripts/qa/ocrbench_extract.py)")
    return _sha256(manifest)


def questions(data: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    subset = load_subset([r["question"] for r in rows])  # q_sha256 check against the committed subset
    return [rows[i] for i in subset]


def run(args: argparse.Namespace) -> int:
    from vlm_pipeline.serve.vllm_runner import run_vllm_batch

    root = project_root()
    data = root / DATA
    manifest_sha = verify_manifest(data)
    qs = questions(data)
    requests = [
        {"id": str(q["row"]), "image": str(data / q["image"]), "prompt": q["question"], "max_tokens": MAX_TOKENS}
        for q in qs
    ]
    res = run_vllm_batch(
        root,
        Path(args.model),
        requests,
        who="bao-tester",
        tag=f"ocrbench-{args.label}",
        quantization=args.quantization,
        min_pixels=PIXELS[0],
        max_pixels=PIXELS[1],
        enable_thinking=False,
        kv_cache_gb=args.kv_cache_gb,
    )
    if not res.ok:
        print(f"batch failed: {res.status} {res.error_code} {res.run_dir}", file=sys.stderr)
        return 1
    by_id = {r["id"]: r for r in res.rows()}
    scored = []
    for q in qs:
        row = by_id[str(q["row"])]
        text = row.get("text", "")
        scored.append(
            {
                "row": q["row"],
                "question_type": q["question_type"],
                "dataset": q["dataset"],
                "score": score_one(text, q["answers"], q["dataset"]) if "error" not in row else 0,
                "error": row.get("error"),
                "finish_reason": row.get("finish_reason"),
                "gen_tokens": row.get("gen_tokens"),
                "prompt_ids_sha": row.get("prompt_ids_sha"),
            }
        )
    n = len(scored)
    result = {
        "benchmark": "ocrbench",
        "subset": "regression_500",
        "label": args.label,
        "model_dir": str(Path(args.model).resolve().relative_to(root)),
        "quantization": args.quantization,
        "run_dir": str(res.run_dir.relative_to(root)),
        "engine": res.engine(),
        "data_manifest_sha256": manifest_sha,
        "max_tokens": MAX_TOKENS,
        "accuracy": round(100 * sum(s["score"] for s in scored) / n, 2),
        "truncated_pct": round(100 * sum(s["finish_reason"] == "length" for s in scored) / n, 2),
        "error_pct": round(100 * sum(s["error"] is not None for s in scored) / n, 2),
        # the "contains" rule rewards long answers by chance: report length to see if one side rambles more
        "gen_tokens_mean": round(sum(s["gen_tokens"] or 0 for s in scored) / n, 1),
        "n": n,
        "citation": CITATION,
        "rows": scored,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("label", "accuracy", "truncated_pct", "error_pct", "n", "run_dir")}))
    return 0


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def comparable(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Reasons two runs cannot be paired (empty = comparable): other questions, data copy or engine settings."""
    problems = []
    if [r["row"] for r in a["rows"]] != [r["row"] for r in b["rows"]]:
        problems.append("different questions or order")
    if a["data_manifest_sha256"] != b["data_manifest_sha256"]:
        problems.append("different data copy (manifest)")
    if a["max_tokens"] != b["max_tokens"]:
        problems.append("different max_tokens")
    for key in SAME_ENGINE:
        if a["engine"].get(key) != b["engine"].get(key):
            problems.append(f"engine {key}: {a['engine'].get(key)!r} != {b['engine'].get(key)!r}")
    return problems


def paired_scores(ref: dict[str, Any], cand: dict[str, Any]) -> tuple[list[int], list[int]]:
    """0/1 scores of both runs matched by question row id (never by position); the id sets must be equal."""
    a = {r["row"]: r["score"] for r in ref["rows"]}
    b = {r["row"]: r["score"] for r in cand["rows"]}
    if a.keys() != b.keys():
        raise SystemExit(f"runs cover different questions: {len(a.keys() ^ b.keys())} ids differ")
    ids = sorted(a)
    return [b[i] for i in ids], [a[i] for i in ids]


def compare(args: argparse.Namespace) -> int:
    ref, cand = _load(args.reference), _load(args.candidate)
    problems = comparable(ref, cand)
    if problems:
        raise SystemExit("not comparable: " + "; ".join(problems))
    pa = paired_accuracy(*paired_scores(ref, cand))
    # gates.yaml thresholds are points on 0-100 (fp8 pass_min_mean -1.0): a 0-1 delta would pass everything.
    if not -100 <= pa["delta"] <= 100 or (pa["delta"] != 0 and abs(pa["delta"]) < 100 / pa["n"] - 1e-9):
        raise SystemExit(f"delta {pa['delta']} is not on the 0-100 points scale")
    report: dict[str, Any] = {"reference": ref["label"], "candidate": cand["label"], "paired": pa}
    if args.noise:
        other = _load(args.noise)
        if comparable(ref, other):
            raise SystemExit("noise run not comparable with the reference")
        report["noise"] = paired_accuracy(*paired_scores(ref, other))
        report["noise"]["changed_answers"] = sum(
            x["score"] != y["score"] for x, y in zip(ref["rows"], other["rows"], strict=True)
        )
    gate = quant_gate(
        [
            BenchmarkResult(
                benchmark="ocrbench",
                n=pa["n"],
                truncated_pct=cand["truncated_pct"],
                delta=pa["delta"],
                delta_ci95=(pa["ci95"][0], pa["ci95"][1]),
            )
        ],
        args.precision,
    )
    report["gate"] = gate.to_json()
    report["accuracy"] = {ref["label"]: ref["accuracy"], cand["label"]: cand["accuracy"]}
    report["gen_tokens_mean"] = {r["label"]: r.get("gen_tokens_mean") for r in (ref, cand)}
    report["truncated_pct"] = {r["label"]: r.get("truncated_pct") for r in (ref, cand)}
    # Preemption recomputes a sequence under another batch: report it; > 0 means the KV pin was too small.
    report["num_preemptions"] = {r["label"]: r["engine"].get("num_preemptions") for r in (ref, cand)}
    print(json.dumps(report, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--quantization", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--kv-cache-gb", type=float, default=KV_CACHE_GB)
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--precision", required=True)
    c.add_argument("--noise", default=None, help="a second run of the reference model (engine noise floor)")
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
