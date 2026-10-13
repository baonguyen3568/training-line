"""Score CORD prediction runs (eval_cord_worker.py output) on the host, with provenance for every number.

  uv run python scripts/qa/score_cord_runs.py label=<run_dir> [label=<run_dir> ...] \
      [--paired a:b ...] [--same a:b ...] [--bootstrap 1000] [--allow-partial] [--out result.json]

--paired a:b  paired bootstrap 95% CI of field F1(b) - F1(a), with the observed difference it belongs to
--same a:b    determinism check: which samples produced a different sequence in the two runs
Comparisons refuse runs that are not on the same images (split and ground truth must match per index) and, unless
--allow-partial, runs whose image sets differ. A run whose records disagree on split, decoding or batch size is
refused: its metrics would mix two settings. The result carries this script's and cord.py's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from vlm_pipeline.eval import cord

MIN_RESAMPLES = 200  # a 2.5% tail needs a handful of resamples to mean anything


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Prediction records keyed by dataset index; refuses duplicates (a run scored twice would inflate n)."""
    records: dict[int, dict[str, Any]] = {}
    for line in (run_dir / "pred.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["index"] in records:
            raise SystemExit(f"{run_dir}: index {record['index']} appears twice")
        records[record["index"]] = record
    if not records:
        raise SystemExit(f"{run_dir}: no predictions")
    return records


def _distinct(rows: list[dict[str, Any]], key: str) -> list[Any]:
    """Distinct values of one record field; old records without it count as "unrecorded"."""
    seen = {json.dumps(r.get(key, "unrecorded"), sort_keys=True) for r in rows}
    return [json.loads(s) for s in sorted(seen)]


def run_settings(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Split / decoding / batch size of a run; a run that mixes settings has no single meaning."""
    settings: dict[str, Any] = {}
    for key in ("split", "decoding", "batch_size"):
        values = _distinct(rows, key)
        if len(values) > 1:
            raise SystemExit(f"run {label!r} mixes {len(values)} values of {key}: {values}")
        settings[key] = values[0]
    settings["worker_sha256"] = _distinct(rows, "worker_sha256")  # may legitimately span commits; reported
    return settings


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = cord.score_rows(records)  # Donut tokens (Parse) or JSON text (chat models), per answer_format
    return {
        "field_f1": round(scores.field_f1, 2),
        "json_exact": round(scores.json_exact, 2),
        "nted_accuracy": round(scores.nted_accuracy, 2),
        "parse_fail_rate": round(scores.parse_fail_rate, 2),
        "truncated_rate": round(100 * sum(bool(r["truncated"]) for r in records) / len(records), 2),
        "n": len(records),
    }


def common_images(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], *, allow_partial: bool
) -> tuple[list[int], dict[str, int]]:
    """Indices both runs scored, after checking they are the same images (same split and ground truth)."""
    ids = sorted(a.keys() & b.keys())
    if not ids:
        raise SystemExit("the two runs share no image")
    for i in ids:
        if a[i].get("split") != b[i].get("split") or a[i]["gt_parse"] != b[i]["gt_parse"]:
            raise SystemExit(f"index {i} is not the same image in both runs (split or ground truth differs)")
    coverage = {"n_common": len(ids), "n_only_a": len(a.keys() - b.keys()), "n_only_b": len(b.keys() - a.keys())}
    if not allow_partial and (coverage["n_only_a"] or coverage["n_only_b"]):
        raise SystemExit(f"runs cover different images {coverage}; pass --allow-partial to compare the overlap")
    return ids, coverage


def _f1(run: dict[int, dict[str, Any]], ids: list[int]) -> float:
    return cord.score_rows([run[i] for i in ids]).field_f1


def paired_bootstrap(
    a: dict[int, dict[str, Any]],
    b: dict[int, dict[str, Any]],
    n_resamples: int,
    *,
    seed: int = 0,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Observed F1(b) - F1(a) on the common images and its 95% percentile CI, resampling images."""
    ids, coverage = common_images(a, b, allow_partial=allow_partial)
    rng = random.Random(seed)  # noqa: S311 - statistics, seeded for reproducibility
    diffs = []
    for _ in range(n_resamples):
        sample = [rng.choice(ids) for _ in ids]
        diffs.append(_f1(b, sample) - _f1(a, sample))
    diffs.sort()
    return {
        **coverage,
        "observed": round(_f1(b, ids) - _f1(a, ids), 2),
        "ci95": [round(diffs[int(0.025 * n_resamples)], 2), round(diffs[int(0.975 * n_resamples) - 1], 2)],
        "resamples": n_resamples,
        "seed": seed,
    }


def different_sequences(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], *, allow_partial: bool = False
) -> dict[str, Any]:
    ids, coverage = common_images(a, b, allow_partial=allow_partial)
    return {**coverage, "different": [i for i in ids if a[i]["sequence"] != b[i]["sequence"]]}


def _pair(text: str) -> tuple[str, str]:
    left, sep, right = text.partition(":")
    if not sep or not left or not right:
        raise argparse.ArgumentTypeError(f"expected a:b, got {text!r}")
    return left, right


def _runs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"expected label=<run_dir>, got {spec!r}")
        if label in runs:
            raise SystemExit(f"label {label!r} given twice")
        runs[label] = Path(path)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=<run_dir>")
    ap.add_argument("--paired", type=_pair, action="append", default=[])
    ap.add_argument("--same", type=_pair, action="append", default=[])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.bootstrap < MIN_RESAMPLES:
        raise SystemExit(f"--bootstrap must be >= {MIN_RESAMPLES}")
    runs = _runs(args.runs)
    unknown = {x for pair in args.paired + args.same for x in pair} - runs.keys()
    if unknown:
        raise SystemExit(f"--paired/--same name unknown labels: {sorted(unknown)}")

    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    result: dict[str, Any] = {
        "scorer_sha256": _sha256(Path(__file__)),
        "cord_py_sha256": _sha256(Path(cord.__file__)),
        "runs": {},
    }
    for label, run_dir in runs.items():
        records = load_run(run_dir)
        loaded[label] = records
        rows = [records[i] for i in sorted(records)]
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        result["runs"][label] = {
            "run_id": info["run_id"],
            "status": info["status"],
            "container_digest": info.get("container_digest"),
            **metrics(rows),
            **run_settings(rows, label),
        }
    result["paired_bootstrap_f1"] = {
        f"{b}-{a}": paired_bootstrap(loaded[a], loaded[b], args.bootstrap, allow_partial=args.allow_partial)
        for a, b in args.paired
    }
    result["different_sequences"] = {
        f"{a}~{b}": different_sequences(loaded[a], loaded[b], allow_partial=args.allow_partial) for a, b in args.same
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
