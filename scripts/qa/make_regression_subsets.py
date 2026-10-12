"""Build the fixed regression subsets of ADR-002 (500 samples x 6 benchmarks), stratified, seeded.

Reads only id/category/question columns of the pinned HF parquet files (no images). Output: one JSON per
benchmark in configs/eval/subsets/regression_500/ with repo, revision, split, seed, method and the sample keys.
Sample key = the dataset's own id when it has one, else the row index in the pinned revision (the order the
harness iterates) plus a short sha256 of the question to detect reordering.
Usage: uv run --with huggingface_hub --with pyarrow python scripts/qa/make_regression_subsets.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

N = 500
SEED = 20261012
OUT = Path("configs/eval/subsets/regression_500")

SPECS: dict[str, dict[str, Any]] = {
    "mmmu_val": {
        "repo": "lmms-lab/MMMU",
        "rev": "364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        "glob": "data/validation-*.parquet",
        "cols": ["id"],
        "id": "id",
        "stratum": lambda r: r["id"].split("_", 1)[1].rsplit("_", 1)[0],
    },
    "mmbench_en_dev": {
        "repo": "lmms-lab/MMBench",
        "rev": "56ba1af8954932c4804bd3f522e05ed96e63b654",
        "glob": "en/dev-*.parquet",
        "cols": ["index", "category"],
        "id": "index",
        "stratum": lambda r: r["category"],
        "group": lambda r: int(r["index"]) % 1_000_000,
    },
    "docvqa_val": {
        "repo": "lmms-lab/DocVQA",
        "rev": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "glob": "DocVQA/validation-*.parquet",
        "cols": ["questionId", "question_types"],
        "id": "questionId",
        "stratum": lambda r: (r["question_types"] or ["none"])[0],
    },
    "chartqa_test": {
        "repo": "lmms-lab/ChartQA",
        "rev": "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "glob": "data/test-*.parquet",
        "cols": ["type", "question"],
        "id": None,
        "stratum": lambda r: r["type"],
    },
    "ocrbench": {
        "repo": "echo840/OCRBench",
        "rev": "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "glob": "data/test-*.parquet",
        "cols": ["question_type", "question"],
        "id": None,
        "stratum": lambda r: r["question_type"],
    },
    "ai2d_test": {
        "repo": "lmms-lab/ai2d",
        "rev": "c83a9b9692933aff8349157c88a413df9d02c4e5",
        "glob": "data/test-*.parquet",
        "cols": ["question"],
        "id": None,
        "stratum": lambda r: "all",
    },
}


def _rows(fs: HfFileSystem, spec: dict[str, Any]) -> list[dict[str, Any]]:
    files = sorted(fs.glob(f"datasets/{spec['repo']}@{spec['rev']}/{spec['glob']}"))
    rows: list[dict[str, Any]] = []
    for f in files:
        with fs.open(f) as fh:
            rows.extend(pq.read_table(fh, columns=spec["cols"]).to_pylist())
    for i, r in enumerate(rows):
        r["_row"] = i
    return rows


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with largest remainder; every stratum gets >= 1 when possible."""
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], max(1, int(x))) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    while sum(alloc.values()) > n:  # too many strata forced to 1: trim the largest
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    return alloc


def build(name: str, spec: dict[str, Any], fs: HfFileSystem) -> dict[str, Any]:
    rows = _rows(fs, spec)
    group_of = spec.get("group", lambda r: r["_row"])
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_of(r)].append(r)
    strata: dict[str, list[Any]] = defaultdict(list)
    for g, members in groups.items():
        strata[str(spec["stratum"](members[0]))].append(g)
    rng = random.Random(f"{SEED}:{name}")  # noqa: S311 - reproducible sampling, not security
    alloc = _allocate({k: len(v) for k, v in strata.items()}, N)
    chosen = sorted(g for k in sorted(strata) for g in rng.sample(sorted(strata[k]), alloc[k]))
    keys = []
    for g in chosen:
        for r in groups[g]:
            key: dict[str, Any] = {"row": r["_row"]}
            if spec["id"]:
                key["id"] = r[spec["id"]]
            if "question" in r:
                key["q_sha256"] = hashlib.sha256(r["question"].encode()).hexdigest()[:16]
            keys.append(key)
    return {
        "benchmark": name,
        "repo": spec["repo"],
        "revision": spec["rev"],
        "split_glob": spec["glob"],
        "seed": SEED,
        "n_questions": len(chosen),
        "n_rows": len(keys),
        "population_rows": len(rows),
        "method": "stratified proportional (largest remainder), groups kept whole"
        + (" (MMBench circular variants share index % 1e6)" if "group" in spec else ""),
        "strata": {k: {"population": len(strata[k]), "sampled": alloc[k]} for k in sorted(strata)},
        "samples": keys,
    }


def main() -> None:
    fs = HfFileSystem()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in SPECS.items():
        result = build(name, spec, fs)
        (OUT / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        print(name, result["n_questions"], "questions /", result["n_rows"], "rows of", result["population_rows"])


if __name__ == "__main__":
    main()
