"""ADR-021 / P021 H5: a downloaded HF split becomes a local dataset folder the local-folder checks accept."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.data.local_folder import scan_folder
from vlm_pipeline.download import convert as cv


def _img(fmt: str = "PNG", size: tuple[int, int] = (64, 48)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, format=fmt)
    return buf.getvalue()


def _run(tmp_path: Path, files: dict[str, bytes], spec: dict[str, Any]) -> tuple[int, Path]:
    src = tmp_path / "files"
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
    (tmp_path / "spec.json").write_text(json.dumps({"limit": 0, "max_bytes": 1 << 30, **spec}))
    out = tmp_path / "out"
    out.mkdir()
    rc = cv.main(["--files", str(src), "--spec", str(tmp_path / "spec.json"), "--out", str(out)])
    return rc, out


def test_an_imagefolder_split_becomes_a_folder_local_checks_accept(tmp_path: Path) -> None:
    meta = [{"file_name": "a.png", "question": "What?\x00", "answer": "A"},
            {"file_name": "b.jpg", "question": "Who?", "answer": ["B", "b"]},
            {"file_name": "../../x.png", "question": "q", "answer": "a"},
            {"file_name": "a.png", "question": "q", "answer": ""}]  # fmt: skip
    files = {"train/metadata.jsonl": "\n".join(json.dumps(m) for m in meta).encode(), "train/a.png": _img(),
             "train/b.jpg": _img("JPEG")}  # fmt: skip
    (tmp_path / "x.png").write_bytes(_img())
    spec = {"files": ["train/metadata.jsonl"], "format": "imagefolder",
            "mapping": {"image": "file_name", "question": "question", "answer": "answer"}}  # fmt: skip
    rc, out = _run(tmp_path, files, spec)
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert (report["rows"], report["skipped"]) == (2, 2)
    assert report["reasons"] == {"image_missing": 1, "empty_target": 1}  # outside the download: never read
    labels = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert labels == [{"image": "images/0.png", "target": "A", "prompt": "What?"},
                      {"image": "images/1.jpg", "target": "B", "prompt": "Who?"}]  # fmt: skip
    scan = scan_folder(out)  # the very checks data.local_folder runs (DATA-V)
    assert (scan.problem_count, len(scan.records)) == (0, 2)


def test_messages_become_prompt_and_target(tmp_path: Path) -> None:
    row = {"image": {"bytes": None, "path": "train/a.png"},
           "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read it"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "TOTAL 5"}]}]}  # fmt: skip
    files = {"train.jsonl": json.dumps(row).encode(), "train/a.png": _img()}
    spec = {"files": ["train.jsonl"], "format": "jsonl", "mapping": {"image": "image", "messages": "messages"}}
    rc, out = _run(tmp_path, files, spec)
    assert rc == 0
    assert json.loads((out / "labels.jsonl").read_text()) == {"image": "images/0.png", "target": "TOTAL 5",
                                                              "prompt": "Read it"}  # fmt: skip


def test_the_limit_and_the_budget_are_kept(tmp_path: Path) -> None:
    rows = [{"file_name": "a.png", "question": "q", "answer": str(n)} for n in range(5)]
    files = {"m/metadata.jsonl": "\n".join(json.dumps(r) for r in rows).encode(), "m/a.png": _img()}
    spec = {"files": ["m/metadata.jsonl"], "format": "imagefolder",
            "mapping": {"image": "file_name", "question": "question", "answer": "answer"}}  # fmt: skip
    rc, out = _run(tmp_path, files, {**spec, "limit": 3})
    assert (rc, json.loads((out / "report.json").read_text())["rows"]) == (0, 3)
    rc, out = _run(tmp_path / "b", files, {**spec, "max_bytes": 10})
    assert (rc, json.loads((out / "report.json").read_text())) == (4, {"error": "over_budget"})


@pytest.mark.parametrize("name", ["../outside.jsonl", "/etc/passwd"])
def test_a_file_outside_the_download_is_refused(tmp_path: Path, name: str) -> None:
    (tmp_path / "outside.jsonl").write_text("{}")
    rc, _ = _run(tmp_path, {"ok.jsonl": b"{}"}, {"files": [name], "format": "jsonl", "mapping": {"answer": "a"}})
    assert rc == 3


def test_a_broken_file_stops_with_a_reason(tmp_path: Path) -> None:
    rc, out = _run(tmp_path, {"t.jsonl": b"{not json"}, {"files": ["t.jsonl"], "format": "jsonl",
                                                          "mapping": {"answer": "a"}})  # fmt: skip
    assert (rc, json.loads((out / "report.json").read_text())) == (4, {"error": "unreadable"})


def test_a_huge_line_is_refused_without_reading_it_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv, "MAX_LINE_BYTES", 64)
    rc, out = _run(tmp_path, {"t.jsonl": b'{"a": "' + b"x" * 500}, {"files": ["t.jsonl"], "format": "jsonl",
                                                                   "mapping": {"answer": "a"}})  # fmt: skip
    assert (rc, json.loads((out / "report.json").read_text())) == (4, {"error": "line_too_long"})


def test_the_shared_question_fills_only_the_rows_without_one(tmp_path: Path) -> None:
    """design 2026-10-14: a user's shared question (e.g. Vietnamese) instead of the trainer's English default."""
    meta = [{"file_name": "a.png", "question": "", "answer": "A"},
            {"file_name": "a.png", "question": "Own?", "answer": "B"}]  # fmt: skip
    files = {"t/metadata.jsonl": "\n".join(json.dumps(m) for m in meta).encode(), "t/a.png": _img()}
    spec = {"files": ["t/metadata.jsonl"], "format": "imagefolder", "prompt": "Đọc hoá đơn này\x07",
            "mapping": {"image": "file_name", "question": "question", "answer": "answer"}}  # fmt: skip
    rc, out = _run(tmp_path, files, spec)
    assert rc == 0
    prompts = [json.loads(x).get("prompt") for x in (out / "labels.jsonl").read_text().splitlines()]
    assert prompts == ["Đọc hoá đơn này", "Own?"]  # control characters dropped; a row's own question wins
