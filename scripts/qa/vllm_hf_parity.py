"""Parity gate vLLM <-> HF for "Chạy thử" on vLLM (B7-3): a catalog model joins VLLM_PARITY_MODELS only on PASS.

Same CORD-v2 test images, same training prompt and pixels, greedy. HF side: scripts/qa/eval_cord_chat_worker.py
(pred.jsonl with prompt_ids_sha + token_ids); vLLM side: ``run`` here (serve.vllm_runner.run_vllm_batch, logprobs=5).

  uv run python scripts/qa/vllm_hf_parity.py run --model <dir> --out outputs/tester/parity/<id>-vllm.json [--n 100]
  uv run python scripts/qa/vllm_hf_parity.py compare <hf pred.jsonl> <vllm.json> [--out report.json]
  uv run python scripts/qa/vllm_hf_parity.py gate <base report.json> <fine-tune report.json>

A model joins only when BOTH its base and a fine-tune of it pass (``gate``); each ``compare`` reports, per run:
1. the same input on 100% of the images - blocking: image_sha256 equal (both sides read the same PNG; the HF worker
   records it) AND prompt_ids_sha equal (same tokens in, image tokens included); a vLLM image error is a failure;
2. no anomaly; 3. real divergences <= 5% of the images. At each image's first differing token, gap = vLLM's logprob
   of the token it chose - its logprob of HF's token (vLLM's top-5 recorded): a near tie = gap < 0.5 nats; an
   anomaly = HF's token scored higher than vLLM's own greedy choice, or that choice is missing from its top-5; a real
   divergence = gap >= 0.5 or HF's token outside the top-5. Rank and the top1-top2 margin are reported only;
4. |delta field F1| <= 2 points AND its paired 95% CI contains 0, and 5. truncated% and parse_fail% each differ by
   <= 2 points. Condition 4 only means something when the model answers in CORD's schema ("quality_informative": HF
   field F1 >= 20), so ``gate`` takes it from the fine-tune; the base must pass 1, 2 and 5.
History (bao-fixcode + bao-tester, 2026-10-13/14), fixed BEFORE the runs it judges: the first rule (exact match >= 90%)
measured answer length - Qwen3.5-2B base, 100 images: same input 100%, exact 53%, 42 of 47 first divergences near
ties (bf16 logprobs move in 0.125 steps; greedy forks at the first tie), and field F1 was 0 on both sides, so the F1
condition could not catch anything on a base. The 5% bound was chosen after that base run showed 5/100 real
divergences (two at gaps 0.5 / 0.625, three outside a top-2 that was then the only one recorded): it is judged on
the R2 fine-tune only: a base rerun with top-5 is the same model, images and greedy decoding, so it would show the
same divergences again (only ranked better), and judging it against a bound read off it would be circular. On the
base, condition 3 is reported, not decisive. The 0.5 nats threshold is unchanged.
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
EXACT_TIE_NATS = 1e-6
LOGPROBS = 5
MAX_REAL_DIVERGENCE_PCT = 5.0
INFORMATIVE_F1 = 20.0
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
        max_pixels=PIXELS[1], enable_thinking=False, logprobs=args.logprobs, kv_cache_gb=args.kv_cache_gb,
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
    """Rank of HF's token among vLLM's recorded top tokens at ``index``: "1".."k" or ">k" (None if unknown)."""
    top = row.get("top_logprobs")
    if hf_token is None or not top or index >= len(top):
        return None
    ranked = [int(t[0]) for t in top[index]]
    return str(ranked.index(hf_token) + 1) if hf_token in ranked else f">{len(ranked)}"


def gap_at(row: dict[str, Any], index: int, hf_token: int | None) -> tuple[float | None, bool]:
    """(vLLM logprob of its own token - of HF's token, or None when HF's is not recorded; own token recorded?)."""
    top = row.get("top_logprobs")
    ids = row.get("token_ids") or []
    if not top or index >= len(top) or index >= len(ids):
        return None, False
    lp = {int(t[0]): float(t[1]) for t in top[index]}
    own = lp.get(int(ids[index]))
    if own is None:
        return None, False
    return (None if hf_token is None or hf_token not in lp else own - lp[hf_token]), True


def kind_of(gap: float | None, own_recorded: bool) -> str:
    """First divergence kind: "near_tie", "anomaly" (greedy did not take its best token) or "divergence"."""
    if not own_recorded or (gap is not None and gap < -EXACT_TIE_NATS):
        return "anomaly"
    if gap is None:
        return "divergence"  # HF's token is outside vLLM's recorded top-k
    return "near_tie" if gap < NEAR_TIE_NATS else "divergence"


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
        h_full = hf[i].get("token_ids") or []
        h_ids, v_ids = _strip_end(h_full, drop), _strip_end(v_ids, drop)
        pos = first_difference(h_ids, v_ids)
        m = margin_at(vl[i], pos) if pos is not None else None
        # HF's token there, its EOS included (HF stopping where vLLM went on is a choice like any other)
        hf_tok = h_full[pos] if pos is not None and pos < len(h_full) else None
        rank = hf_rank(vl[i], pos, hf_tok) if pos is not None else None
        gap, own = gap_at(vl[i], pos, hf_tok) if pos is not None else (None, False)
        diverged.append(
            {"id": i, "first_diff_token": pos, "hf_choice_rank_in_vllm": rank,
             "vllm_margin_nats": None if m is None else round(m, 3),
             "gap_nats": None if gap is None else round(gap, 3), "kind": kind_of(gap, own)}
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
    kinds = [d["kind"] for d in diverged]
    near = sorted(d["gap_nats"] for d in diverged if d["kind"] == "near_tie")
    real_pct = 100 * kinds.count("divergence") / n
    informative = hf_scores.field_f1 >= INFORMATIVE_F1
    checks = {
        "1_same_input": len(same_input) == n,
        "2_no_anomaly": "anomaly" not in kinds,
        "3_real_divergences_le_5pct": real_pct <= MAX_REAL_DIVERGENCE_PCT,
        "4_f1_delta_le_2_and_ci_has_0": abs(delta) <= 2 and ci[0] <= 0 <= ci[1],
        "5_trunc_and_parse_fail_le_2": abs(trunc_hf - trunc_vl) <= 2
        and abs(hf_scores.parse_fail_rate - vl_scores.parse_fail_rate) <= 2,
    }
    exact_ge_90 = 100 * len(exact) / n >= 90
    return {
        "verdict": "PASS" if all(checks.values()) and informative else "FAIL",
        "checks": checks,
        "quality_informative": informative,
        "n": n,
        "same_input_pct": round(100 * len(same_input) / n, 1),
        "vllm_image_errors": image_errors,
        "exact_match_pct": round(100 * len(exact) / n, 1),
        "field_f1": {"hf": round(hf_scores.field_f1, 2), "vllm": round(vl_scores.field_f1, 2)},
        "delta_f1_vllm_minus_hf": {"delta": round(delta, 2), "ci95": [round(ci[0], 2), round(ci[1], 2)]},
        "truncated_pct": {"hf": round(trunc_hf, 1), "vllm": round(trunc_vl, 1)},
        "parse_fail_pct": {"hf": round(hf_scores.parse_fail_rate, 1), "vllm": round(vl_scores.parse_fail_rate, 1)},
        "near_ties": len(near),
        "anomalies": [d["id"] for d in diverged if d["kind"] == "anomaly"],
        "real_divergences": [d["id"] for d in diverged if d["kind"] == "divergence"],
        "real_divergence_pct": round(real_pct, 1),
        "near_tie_gap_nats": {"median": near[len(near) // 2] if near else None, "max": near[-1] if near else None},
        "original_rule": {"2_exact_match_ge_90": exact_ge_90},
        "diverged": diverged,
    }


def gate(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    """Join decision: the base passes 1, 2, 5 and the fine-tune passes everything, its F1 informative.

    The base's condition 3 is reported only: its numbers were seen before the 5% bound was chosen (see above).
    """
    need_base = ("1_same_input", "2_no_anomaly", "5_trunc_and_parse_fail_le_2")
    failed = [f"base:{k}" for k in need_base if not base["checks"].get(k)]
    failed += [f"fine_tune:{k}" for k, ok in tuned["checks"].items() if not ok]
    if not tuned.get("quality_informative"):
        failed.append("fine_tune:quality_informative")
    return {
        "verdict": "FAIL" if failed else "PASS",
        "failed": failed,
        "base_real_divergence_pct": base.get("real_divergence_pct"),
        "base_note": "base: numbers seen before the threshold was chosen; condition 3 reported, not decisive",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--kv-cache-gb", type=float, default=8.0)
    r.add_argument("--logprobs", type=int, default=LOGPROBS)
    c = sub.add_parser("compare")
    c.add_argument("hf")
    c.add_argument("vllm")
    c.add_argument("--out", default=None)
    g = sub.add_parser("gate")
    g.add_argument("base")
    g.add_argument("fine_tune")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "gate":
        verdict = gate(json.loads(Path(args.base).read_text()), json.loads(Path(args.fine_tune).read_text()))
        print(json.dumps(verdict, indent=1))
        return 0 if verdict["verdict"] == "PASS" else 1
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
