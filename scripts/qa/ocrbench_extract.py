"""Extract OCRBench (echo840/OCRBench@92a54bd1, eval_only) into files the vLLM batch can mount.

One image file per row (bytes copied as stored, never decoded here: decoding happens in the job under
common.image_rules) plus ``questions.jsonl`` (row, question, answers, dataset, question_type, q_sha256).
Runs INSIDE a vlm-automodel image (pyarrow), ``--network none``. Security conditions (2026-10-13): local copy under
data/ only (gitignored), nothing leaves the machine, reports publish aggregate scores and cite the paper
(eval.ocrbench.CITATION).

Security (2026-10-13): the parquet must match the catalog's files_sha256 before anything is extracted, and the
output carries ``MANIFEST.sha256`` (sha256sum format: every image + questions.jsonl) that the R3 gate verifies; its
own hash is recorded with each score so a number names exactly one copy of the data.

  python /workspace/scripts/qa/ocrbench_extract.py --parquet <snapshot>/data/test-00000-of-00001.parquet \
      --out /workspace/data/eval/ocrbench-92a54bd1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "echo840/OCRBench"
CATALOG = Path(__file__).resolve().parents[2] / "configs" / "catalog" / "datasets.yaml"
MANIFEST = "MANIFEST.sha256"
MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpg"), (b"GIF8", "gif"), (b"BM", "bmp"))


def extension(data: bytes) -> str:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("unknown image format (not in the allowlist)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256(parquet: Path, catalog: Path = CATALOG) -> str:
    """The catalog's files_sha256 for this parquet (matched on its path inside the snapshot, e.g. data/test-…)."""
    import yaml

    data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    entries = data["datasets"] if isinstance(data, dict) else data
    entry = next(e for e in entries if e.get("repo") == REPO)
    for rel, sha in entry["files_sha256"].items():
        if parquet.as_posix().endswith("/" + rel):
            return str(sha)
    raise SystemExit(f"{parquet}: not listed in the catalog files_sha256 of {REPO}")


def write_manifest(out: Path) -> str:
    """sha256sum-format manifest of every image + questions.jsonl; returns the manifest's own sha256."""
    files = [*sorted((out / "images").iterdir()), out / "questions.jsonl"]
    lines = [f"{sha256_file(f)}  {f.relative_to(out).as_posix()}" for f in files]
    (out / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(out / MANIFEST)


def main() -> int:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]  # container-only dependency

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parquet = Path(args.parquet)
    want, got = expected_sha256(parquet), sha256_file(parquet)
    if want != got:
        raise SystemExit(f"{parquet}: sha256 {got} differs from the catalog {want}; refusing to extract")
    table = pq.read_table(parquet).to_pylist()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with (out / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for row, rec in enumerate(table):
            data = rec["image"]["bytes"]
            name = f"{row:04d}.{extension(data)}"
            (out / "images" / name).write_bytes(data)
            answers = rec["answer"]
            fh.write(
                json.dumps(
                    {
                        "row": row,
                        "image": f"images/{name}",
                        "question": rec["question"],
                        "answers": list(answers) if not isinstance(answers, str) else [answers],
                        "dataset": rec["dataset"],
                        "question_type": rec["question_type"],
                        "q_sha256": hashlib.sha256(rec["question"].encode()).hexdigest()[:16],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"extracted {len(table)} rows to {out}; {MANIFEST} sha256 {write_manifest(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
