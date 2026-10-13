"""Parity gate vLLM <-> HF for "Chạy thử" on vLLM (B7-3): a catalog model joins VLLM_PARITY_MODELS only on PASS.

Same CORD-v2 test images, same training prompt and pixels, greedy. HF side: scripts/qa/eval_cord_chat_worker.py
(pred.jsonl with prompt_ids_sha + token_ids); vLLM side: ``run`` here (serve.vllm_runner.run_vllm_batch, logprobs=2).

  uv run python scripts/qa/vllm_hf_parity.py run --model <dir> --out outputs/tester/parity/<id>-vllm.json [--n 100]
  uv run python scripts/qa/vllm_hf_parity.py compare <hf pred.jsonl> <vllm.json> [--out report.json]

PASS needs all of (agreed with bao-fixcode, 2026-10-13):
1. the same input on 100% of the images - blocking: image_sha256 equal (both sides read the same PNG; the HF worker
   records it) AND prompt_ids_sha equal (same tokens in, image tokens included); a vLLM image error is a failure;
2. exact text match >= 90%; for the others the first differing token, the rank of HF's token in vLLM's top-2 and
   vLLM's top1-top2 margin there (a near tie = HF's token is vLLM's rank 2 with margin < 0.5 nats: a kernel-level
   difference; anything else is a real divergence);
3. |delta field F1| <= 2 points AND its paired 95% CI contains 0;
4. truncated% and parse_fail% each differ by <= 2 points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from vlm_pipeline.common.paths import project_root
from vlm_pipeline.eval import cord

DATA = Path("data") / "reference" / "cord-v2-test"  # research's folder copy of CORD-v2 test (labels.jsonl + PNGs)
PROMPT = "Read this receipt and return its fields as JSON."
PIXELS = (200704, 1003520)
MAX_TOKENS = 1024
NEAR_TIE_NATS = 0.5
RESAMPLES, SEED = 1000, 0


def run(args: argparse.Namespace) -> int:
    from vlm_pipeline.serve.vllm_runner import run_vllm_batch

    root = project_root()
    labels = [json.loads(line) for line in (root / DATA / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [
        {"id": str(i), "image": str(root / DATA / row["image"]), "prompt": PROMPT, "max_tokens": MAX_TOKENS}
        for i, row in enumerate(labels[: args.n])
    ]
    res = run_vllm_batch(
        root, Path(args.model), items, who="bao-tester", tag="parity-vllm", min_pixels=PIXELS[0],
        max_pixels=PIXELS[1], enable_thinking=False, logprobs=2, kv_cache_gb=args.kv_cache_gb,
    )  # fmt: skip
    if not res.ok:
        print(f"batch failed: {res.status} {res.error_code} {res.run_dir}", file=sys.stderr)
        return 1
    shas = {str(it["id"]): hashlib.sha256(Path(str(it["image"])).read_bytes()).hexdigest() for it in items}
    rows = [{**row, "image_sha256": shas.get(row["id"])} for row in res.rows()]
    out = {"run_dir": str(res.run_dir.relative_to(root)), "engine": res.engine(), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": out["run_dir"], "n": len(out["rows"])}))
    return 0


def first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token (or the shorter length when one is a prefix), None when equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def margin_at(row: dict[str, Any], index: int) -> float | None:
    """Top1 - top2 vLLM logprob at ``index`` (nats); None when not recorded."""
    top = row.get("top_logprobs")
    if not top or index >= len(top) or len(top[index]) < 2:
        return None
    return float(top[index][0][1]) - float(top[index][1][1])


def _strip_end(ids: list[int], drop: set[int]) -> list[int]:
    end = len(ids)
    while end and ids[end - 1] in drop:
        end -= 1
    return ids[:end]


def hf_rank(row: dict[str, Any], index: int, hf_token: int | None) -> str | None:
    """Rank of HF's token among vLLM's recorded top tokens at ``index``: "1", "2" or ">2" (None if unknown)."""
    top = row.get("top_logprobs")
    if hf_token is None or not top or index >= len(top):
        return None
    ranked = [int(t[0]) for t in top[index]]
    return str(ranked.index(hf_token) + 1) if hf_token in ranked else ">2"


def compare(hf: dict[int, dict[str, Any]], vl: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(hf)
    if not ids:
        raise SystemExit("no HF predictions to compare")
    if ids != sorted(vl):
        raise SystemExit(f"different images: {len(set(hf) ^ set(vl))} ids differ")
    image_errors = [i for i in ids if "error" in vl[i]]
    same_input = [
        i
        for i in ids
        if i not in image_errors
        and hf[i].get("image_sha256")
        and hf[i]["image_sha256"] == vl[i].get("image_sha256")
        and hf[i].get("prompt_ids_sha")
        and hf[i]["prompt_ids_sha"] == vl[i].get("prompt_ids_sha")
    ]
    text = {i: "" if i in image_errors else vl[i]["text"].strip() for i in ids}
    exact = [i for i in ids if i not in image_errors and hf[i]["sequence"] == text[i]]
    diverged: list[dict[str, Any]] = []
    for i in ids:
        if i in exact or i in image_errors:
            continue
        v_ids = vl[i].get("token_ids") or []
        drop = {v_ids[-1]} if v_ids and vl[i].get("finish_reason") == "stop" else set()  # EOS / pad at the end
        h_ids, v_ids = _strip_end(hf[i].get("token_ids") or [], drop), _strip_end(v_ids, drop)
        pos = first_difference(h_ids, v_ids)
        m = margin_at(vl[i], pos) if pos is not None else None
        rank = hf_rank(vl[i], pos, h_ids[pos] if pos < len(h_ids) else None) if pos is not None else None
        diverged.append(
            {"id": i, "first_diff_token": pos, "hf_choice_rank_in_vllm": rank,
             "vllm_margin_nats": None if m is None else round(m, 3)}
        )  # fmt: skip
    hf_rows = [{"answer_format": "json", "sequence": hf[i]["sequence"], "gt_parse": hf[i]["gt_parse"]} for i in ids]
    vl_rows = [{"answer_format": "json", "sequence": text[i], "gt_parse": hf[i]["gt_parse"]} for i in ids]
    a = [cord.field_f1_counts(cord.prediction_of(r) or {}, r["gt_parse"]) for r in hf_rows]
    b = [cord.field_f1_counts(cord.prediction_of(r) or {}, r["gt_parse"]) for r in vl_rows]
    rng, n, diffs = random.Random(SEED), len(ids), []  # noqa: S311 - statistics, seeded
    for _ in range(RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100 * (cord.f1_from_counts([b[k] for k in idx]) - cord.f1_from_counts([a[k] for k in idx])))
    diffs.sort()
    delta = 100 * (cord.f1_from_counts(b) - cord.f1_from_counts(a))
    ci = [diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES) - 1]]
    hf_scores, vl_scores = cord.score_rows(hf_rows), cord.score_rows(vl_rows)
    trunc_hf = 100 * sum(bool(hf[i]["truncated"]) for i in ids) / n
    trunc_vl = 100 * sum(vl[i].get("finish_reason") == "length" for i in ids) / n
    checks = {
        "1_same_input": len(same_input) == n,
        "2_exact_match_ge_90": 100 * len(exact) / n >= 90,
        "3_f1_delta_le_2_and_ci_has_0": abs(delta) <= 2 and ci[0] <= 0 <= ci[1],
        "4_trunc_and_parse_fail_le_2": abs(trunc_hf - trunc_vl) <= 2
        and abs(hf_scores.parse_fail_rate - vl_scores.parse_fail_rate) <= 2,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n": n,
        "same_input_pct": round(100 * len(same_input) / n, 1),
        "vllm_image_errors": image_errors,
        "exact_match_pct": round(100 * len(exact) / n, 1),
        "field_f1": {"hf": round(hf_scores.field_f1, 2), "vllm": round(vl_scores.field_f1, 2)},
        "delta_f1_vllm_minus_hf": {"delta": round(delta, 2), "ci95": [round(ci[0], 2), round(ci[1], 2)]},
        "truncated_pct": {"hf": round(trunc_hf, 1), "vllm": round(trunc_vl, 1)},
        "parse_fail_pct": {"hf": round(hf_scores.parse_fail_rate, 1), "vllm": round(vl_scores.parse_fail_rate, 1)},
        "near_ties": sum(
            1
            for d in diverged
            if d["hf_choice_rank_in_vllm"] == "2"
            and d["vllm_margin_nats"] is not None
            and d["vllm_margin_nats"] < NEAR_TIE_NATS
        ),
        "diverged": diverged,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--kv-cache-gb", type=float, default=8.0)
    c = sub.add_parser("compare")
    c.add_argument("hf")
    c.add_argument("vllm")
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    hf = {int(json.loads(line)["index"]): json.loads(line) for line in Path(args.hf).read_text().splitlines()}
    vl = {int(row["id"]): row for row in json.loads(Path(args.vllm).read_text())["rows"]}
    report = compare(hf, vl)
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
