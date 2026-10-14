"""Inspect a user-picked Hugging Face dataset sample (ADR-021 / P021 H4): columns, a column mapping, a preview.

``python -m vlm_pipeline.download.inspect --sample <dir> --format <fmt> --out <dir>`` runs as a ``data`` job
(``--network none``, CPU, its run dir the only writable place; security 2026-10-14: pyarrow and the image decoders
are a large parse surface, kept out of the app process). It reads the ONE bounded sample the resolve job fetched and
writes OUR files only:

* ``<out>/inspect.json``: ``{columns: [{name, kind}], suggested: {image?, question?, answer?, messages?},
  samples: [{question?, answer?, thumb?, error?}], rows_read}``;
* ``<out>/thumbs/<n>.png``: images decoded under ``common.image_rules`` and re-encoded to PNG, at most 256 px a
  side, no metadata.

Text is untrusted: control characters dropped, truncated. Nothing from the dataset is ever executed or imported; a
file that does not parse is reported as such, never guessed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MAX_ROWS = 20
MAX_FILE_BYTES = 256 << 20  # = dataset_layout.SAMPLE_MAX
MAX_LINE_BYTES = 1 << 20  # one jsonl line / csv field
MAX_TEXT = 500  # preview text per field
THUMB_SIDE = 256
MAX_COLUMNS = 64
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NAME = re.compile(r"[^A-Za-z0-9_.\- ]")
# Column names the mapping suggests, by role (lowercase, first match wins); the UI shows them for confirmation.
SUGGEST = {
    "messages": ("messages", "conversations", "conversation", "chat"),
    "question": ("question", "query", "prompt", "instruction", "input", "problem"),
    "answer": ("answer", "answers", "response", "output", "label", "ground_truth", "target", "caption", "text"),
    "image": ("image", "images", "img", "picture", "photo"),
}


def clean_text(value: object, limit: int = MAX_TEXT) -> str:
    """Untrusted text for the preview: control characters dropped, whitespace trimmed, truncated."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return _CONTROL.sub("", text).strip()[:limit]


def clean_column(name: object) -> str:
    return _NAME.sub("", str(name))[:64] or "column"


# --------------------------------------------------------------------------------------------- reading
def _regular(path: Path, cap: int) -> None:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
        raise ValueError(f"{path.name}: not a regular file under the size cap")


def read_rows(path: Path, fmt: str) -> list[dict[str, Any]]:
    """The first MAX_ROWS rows of one data file (no symlink, size-capped), as plain Python values."""
    _regular(path, MAX_FILE_BYTES)
    if fmt == "parquet":
        import pyarrow.parquet as pq  # type: ignore[import-not-found]  # job image only

        parquet = pq.ParquetFile(str(path))
        batch = next(parquet.iter_batches(batch_size=MAX_ROWS), None)
        return [] if batch is None else batch.to_pylist()[:MAX_ROWS]
    if fmt == "arrow":
        import pyarrow as pa  # job image only (the import above already marks pyarrow missing for mypy)

        with pa.memory_map(str(path)) as source:
            reader = pa.ipc.open_stream(source)
            batch = next(iter(reader), None)
        return [] if batch is None else batch.to_pylist()[:MAX_ROWS]
    if fmt in ("jsonl", "imagefolder") and path.suffix == ".jsonl":
        rows = []
        with path.open("rb") as fh:
            for line in fh:
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("a json line over the size cap")
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= MAX_ROWS:
                    break
        return [r for r in rows if isinstance(r, dict)]
    if fmt == "json":
        data = json.loads(path.read_bytes())
        items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        return [r for r in items[:MAX_ROWS] if isinstance(r, dict)]
    if fmt in ("csv", "imagefolder"):
        csv.field_size_limit(MAX_LINE_BYTES)
        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(r) for _, r in zip(range(MAX_ROWS), csv.DictReader(fh), strict=False)]
    raise ValueError(f"format {fmt!r} is not read")


# --------------------------------------------------------------------------------------------- columns
def _is_image(value: Any) -> bool:
    if isinstance(value, dict) and ("bytes" in value or "path" in value):  # the hub's Image feature
        return isinstance(value.get("bytes"), bytes | bytearray) or isinstance(value.get("path"), str)
    return isinstance(value, bytes | bytearray) and value[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1",
                                                                 b"RIFF", b"\xff\xd8\xff\xdb")  # fmt: skip


def _is_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(m, dict) and ("role" in m or "from" in m) and ("content" in m or "value" in m) for m in value
        )
    )


def column_kind(values: Iterable[Any], name: str = "") -> str:  # noqa: PLR0911 - one return per kind
    """The kind (image, messages, text, number, list, other) of the first non-empty value (or an image file name)."""
    seen = [v for v in values if v not in (None, "", [], {})]
    if not seen:
        return "other"
    first = seen[0]
    if _is_image(first) or (name.lower() in ("file_name", "image_path") and isinstance(first, str)):
        return "image"
    if _is_messages(first):
        return "messages"
    if isinstance(first, bool):
        return "other"
    if isinstance(first, int | float):
        return "number"
    if isinstance(first, str):
        return "text"
    if isinstance(first, list):
        return "list"
    return "other"


def columns(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    names: list[str] = []
    for row in rows:
        names += [k for k in row if k not in names]
    return [{"name": clean_column(n), "kind": column_kind((r.get(n) for r in rows), n), "raw": n}
            for n in names[:MAX_COLUMNS]]  # fmt: skip


def suggest(cols: list[dict[str, str]]) -> dict[str, str]:
    """Suggested role -> column name: messages wins over question/answer; each column used once; image by kind."""
    out: dict[str, str] = {}
    used: set[str] = set()

    def pick(role: str, kinds: tuple[str, ...]) -> None:
        for wanted in SUGGEST[role]:
            for c in cols:
                if c["name"] not in used and c["kind"] in kinds and c["name"].lower() == wanted:
                    out[role] = c["name"]
                    used.add(c["name"])
                    return

    images = [c for c in cols if c["kind"] == "image"]
    if images:
        named = next((c for c in images if c["name"].lower() in SUGGEST["image"]), images[0])
        out["image"] = named["name"]
        used.add(named["name"])
    pick("messages", ("messages",))
    if "messages" not in out:
        pick("question", ("text",))
        pick("answer", ("text", "list", "number"))
    return out


# --------------------------------------------------------------------------------------------- preview
def _image_bytes(value: Any, sample_dir: Path) -> bytes | None:
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, bytes | bytearray):
            return bytes(data)
        value = value.get("path")
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str):  # imagefolder: a file next to the metadata (never outside the sample dir)
        given = sample_dir / value
        candidate = given.resolve()  # the real block is is_relative_to; a link named by the metadata is refused too
        if not given.is_symlink() and candidate.is_relative_to(sample_dir.resolve()) and candidate.is_file():
            from vlm_pipeline.common.image_rules import MAX_IMAGE_BYTES

            if candidate.stat().st_size <= MAX_IMAGE_BYTES:
                return candidate.read_bytes()
    return None


def write_thumb(data: bytes, out: Path) -> None:
    """Decode under the shared image rules, shrink to THUMB_SIDE, re-encode as a metadata-free PNG."""
    from vlm_pipeline.common.image_rules import open_image_bytes

    img = open_image_bytes(data)
    img.thumbnail((THUMB_SIDE, THUMB_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out.write_bytes(buf.getvalue())


def preview(rows: list[dict[str, Any]], cols: list[dict[str, str]], mapping: Mapping[str, str], sample_dir: Path,
            thumbs: Path) -> list[dict[str, Any]]:  # fmt: skip
    from vlm_pipeline.common.image_rules import ImageRuleError

    raw = {c["name"]: c["raw"] for c in cols}
    samples: list[dict[str, Any]] = []
    for n, row in enumerate(rows):
        item: dict[str, Any] = {}
        if "messages" in mapping:
            msgs = row.get(raw[mapping["messages"]]) or []
            texts = [clean_text(m.get("content", m.get("value", ""))) for m in msgs if isinstance(m, dict)]
            item.update(question=texts[0] if texts else "", answer=texts[1] if len(texts) > 1 else "")
        for role in ("question", "answer"):
            if role in mapping:
                item[role] = clean_text(row.get(raw[mapping[role]], ""))
        if "image" in mapping:
            data = _image_bytes(row.get(raw[mapping["image"]]), sample_dir)
            try:
                if data is None:
                    raise ImageRuleError("missing")
                thumbs.mkdir(parents=True, exist_ok=True)
                write_thumb(data, thumbs / f"{n}.png")
                item["thumb"] = n
            except ImageRuleError as exc:
                item["error"] = f"image_{exc.reason}"
        samples.append(item)
    return samples


def inspect(sample_dir: Path, fmt: str, out: Path) -> dict[str, Any]:
    files = sorted(p for p in sample_dir.rglob("*") if p.is_file() and not p.is_symlink())
    data = next((p for p in files if p.suffix.lower() in (".parquet", ".arrow", ".jsonl", ".json", ".csv")), None)
    if data is None:
        return {"columns": [], "suggested": {}, "samples": [], "rows_read": 0, "error": "no_sample"}
    rows = read_rows(data, fmt)
    base = data.parent if fmt == "imagefolder" else sample_dir
    cols = columns(rows)
    mapping = suggest(cols)
    samples = preview(rows, cols, mapping, base, out / "thumbs")
    return {"columns": [{"name": c["name"], "kind": c["kind"]} for c in cols], "suggested": mapping,
            "samples": samples, "rows_read": len(rows)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vlm_pipeline.download.inspect")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--format", required=True, choices=("parquet", "arrow", "jsonl", "json", "csv", "imagefolder"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        result = inspect(args.sample, args.format, args.out)
    except (OSError, ValueError, UnicodeDecodeError) as exc:  # a file that does not parse: said, never guessed
        print(f"download.inspect: {type(exc).__name__}", file=sys.stderr, flush=True)
        result = {"columns": [], "suggested": {}, "samples": [], "rows_read": 0, "error": "unreadable"}
    (args.out / "inspect.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
