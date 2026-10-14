"""Turn a downloaded Hugging Face dataset split into a local dataset folder (ADR-021 / P021 H5).

``python -m vlm_pipeline.download.convert --files <dir> --spec <json> --out <dir>`` runs as a ``data`` job
(``--network none``, CPU, the downloaded files read-only, its run dir the only writable place). ``--spec`` (written
by the host) names the split's files, their format, the column mapping and the limits. The job streams the rows and
writes the folder ``data.local_folder`` reads, so the very same checks run on it (DATA-V):

* ``<out>/images/<n>.png|jpg``: every image decoded under ``common.image_rules`` and re-encoded (JPEG stays JPEG,
  anything else becomes PNG); never a file of the dataset under its own name;
* ``<out>/labels.jsonl``: ``{"image": "images/<n>.<ext>", "prompt": <question>, "target": <answer>}`` per row;
* ``<out>/report.json``: ``{rows, skipped, reasons, bytes}``.

A row whose image or text cannot be used is skipped and counted (never guessed); a file that does not parse stops
the job with a reason. Text is untrusted: control characters dropped, lengths capped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import stat
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

MAX_ROWS = 2_000_000  # security checklist §10: whole dataset
MAX_PROMPT = 2000  # = data.local_folder.MAX_PROMPT_CHARS
MAX_TARGET = 16_000
MAX_LINE_BYTES = 16 << 20
MAX_FILE_BYTES = 8 << 30
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_JPEG = b"\xff\xd8\xff"


class ConvertError(ValueError):
    """The split cannot be converted (a file does not parse, or a limit is crossed); ``reason`` says which."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def clean(value: object, limit: int) -> str:
    text = value if isinstance(value, str) else "" if value is None else json.dumps(value, ensure_ascii=False)
    return _CONTROL.sub("", text).strip()[:limit]


# --------------------------------------------------------------------------------------------- rows
def _regular(path: Path) -> None:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE_BYTES:
        raise ConvertError("not_a_regular_file")


def iter_rows(path: Path, fmt: str) -> Iterator[dict[str, Any]]:  # noqa: PLR0912 - one branch per format
    """Every row of one data file, streamed (parquet by batch, json lines one by one)."""
    _regular(path)
    if fmt == "parquet":
        import pyarrow.parquet as pq  # type: ignore[import-not-found]  # job image only

        for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif fmt == "arrow":
        import pyarrow as pa  # job image only (the import above already marks pyarrow missing for mypy)

        with pa.memory_map(str(path)) as source:
            for batch in pa.ipc.open_stream(source):
                yield from batch.to_pylist()
    elif fmt in ("jsonl", "imagefolder") and path.suffix == ".jsonl":
        with path.open("rb") as fh:
            while line := fh.readline(MAX_LINE_BYTES + 1):  # bounded: a 6 GB line never lands in memory
                if len(line) > MAX_LINE_BYTES:
                    raise ConvertError("line_too_long")
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
    elif fmt == "webdataset":  # streamed under the archive rules; a refused shard stops the job (security §10)
        from vlm_pipeline.download.archives import ArchiveRefused, ShardReport, iter_samples, sample_to_row

        try:
            for sample in iter_samples([path], ShardReport()):
                yield sample_to_row(sample)
        except ArchiveRefused as exc:
            raise ConvertError(f"shard_{exc.reason}") from None
    elif fmt == "json":
        data = json.loads(path.read_bytes())
        items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        yield from (r for r in items if isinstance(r, dict))
    elif fmt in ("csv", "imagefolder"):
        csv.field_size_limit(MAX_LINE_BYTES)
        with path.open(newline="", encoding="utf-8") as fh:
            yield from (dict(r) for r in csv.DictReader(fh))
    else:
        raise ConvertError("unknown_format")


def _image_bytes(value: Any, base: Path) -> bytes | None:
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, bytes | bytearray):
            return bytes(data)
        value = value.get("path")
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, str):  # imagefolder: next to its metadata, never outside the downloaded files
        given = base / value
        candidate = given.resolve()
        if not given.is_symlink() and candidate.is_relative_to(base.resolve()) and candidate.is_file():
            from vlm_pipeline.common.image_rules import MAX_IMAGE_BYTES

            if candidate.stat().st_size <= MAX_IMAGE_BYTES:
                return candidate.read_bytes()
    return None


def _texts(row: Mapping[str, Any], mapping: Mapping[str, str]) -> tuple[str, str]:
    """(prompt, target) of one row: a messages column (first user turn, first assistant turn) or two columns."""
    if "messages" in mapping:
        prompt = target = ""
        for m in row.get(mapping["messages"]) or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", m.get("from", ""))).lower()
            content = m.get("content", m.get("value", ""))
            if isinstance(content, list):  # HF chat: [{"type": "text", "text": ...}, {"type": "image"}]
                content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
            if role in ("user", "human") and not prompt:
                prompt = clean(content, MAX_PROMPT)
            elif role in ("assistant", "gpt") and not target:
                target = clean(content, MAX_TARGET)
        return prompt, target
    answer = row.get(mapping["answer"]) if "answer" in mapping else None
    if isinstance(answer, list) and answer:  # VQA sets list several valid answers: the first is the target
        answer = answer[0]
    question = row.get(mapping["question"]) if "question" in mapping else ""
    return clean(question, MAX_PROMPT), clean(answer, MAX_TARGET)


# --------------------------------------------------------------------------------------------- convert
def convert(files: list[Path], fmt: str, mapping: Mapping[str, str], out: Path, *, limit: int,
            max_bytes: int, default_prompt: str = "") -> dict[str, Any]:  # fmt: skip
    from vlm_pipeline.common.image_rules import ImageRuleError, open_image_bytes

    images = out / "images"
    images.mkdir(parents=True)
    skipped: Counter[str] = Counter()
    rows = written = 0
    with (out / "labels.jsonl").open("w", encoding="utf-8") as labels:
        for path in files:
            for row in iter_rows(path, fmt):
                if (limit and rows >= limit) or rows >= MAX_ROWS:
                    break
                prompt, target = _texts(row, mapping)
                prompt = prompt or default_prompt  # the user's shared question; none: the trainer's DEFAULT_PROMPT
                if not target:
                    skipped["empty_target"] += 1
                    continue
                data = _image_bytes(row.get(mapping["image"]), path.parent) if "image" in mapping else None
                if data is None:
                    skipped["image_missing"] += 1
                    continue
                try:
                    img = open_image_bytes(data)
                except ImageRuleError as exc:
                    skipped[f"image_{exc.reason}"] += 1
                    continue
                ext = "jpg" if data[:3] == _JPEG else "png"
                buf = io.BytesIO()
                img.save(buf, format="JPEG" if ext == "jpg" else "PNG", **({"quality": 95} if ext == "jpg" else {}))
                written += buf.tell()
                if written > max_bytes:
                    raise ConvertError("over_budget")
                name = f"images/{rows}.{ext}"
                (out / name).write_bytes(buf.getvalue())
                item = {"image": name, "target": target, **({"prompt": prompt} if prompt else {})}
                labels.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
    return {"rows": rows, "skipped": sum(skipped.values()), "reasons": dict(skipped), "bytes": written}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vlm_pipeline.download.convert")
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    base = args.files.resolve()
    files = []
    for name in spec["files"]:  # the host's list (checked names), each inside the read-only files dir
        path = (base / name).resolve()
        if not path.is_relative_to(base) or (base / name).is_symlink():
            print("download.convert: a file outside the download", file=sys.stderr)
            return 3
        files.append(path)
    try:
        report = convert(files, spec["format"], spec["mapping"], args.out, limit=int(spec.get("limit", 0)),
                         max_bytes=int(spec["max_bytes"]),
                         default_prompt=clean(spec.get("prompt"), MAX_PROMPT))  # fmt: skip
    except (ConvertError, OSError, ValueError, UnicodeDecodeError) as exc:
        reason = exc.reason if isinstance(exc, ConvertError) else "unreadable"
        (args.out / "report.json").write_text(json.dumps({"error": reason}), encoding="utf-8")
        print(f"download.convert: {reason}", file=sys.stderr)
        return 4
    (args.out / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
