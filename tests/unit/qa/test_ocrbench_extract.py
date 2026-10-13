"""OCRBench extraction guards (security conditions 2026-10-13): source pinned by the catalog sha, output manifested."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "ocrbench_extract.py"


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("ocrbench_extract", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pinned_parquet_sha_comes_from_the_catalog() -> None:
    mod = _mod()
    parquet = Path("/x/snapshots/rev/data/test-00000-of-00001.parquet")
    assert mod.expected_sha256(parquet).startswith("d9cc9f48")
    with pytest.raises(SystemExit, match="not listed"):
        mod.expected_sha256(Path("/x/data/other.parquet"))


def test_manifest_lists_every_file_and_detects_a_change(tmp_path: Path) -> None:
    mod = _mod()
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff a")
    (tmp_path / "questions.jsonl").write_text('{"row": 0}\n')
    first = mod.write_manifest(tmp_path)
    lines = (tmp_path / "MANIFEST.sha256").read_text().splitlines()
    assert lines == [
        f"{hashlib.sha256(b'\xff\xd8\xff a').hexdigest()}  images/0000.jpg",
        f"{hashlib.sha256(b'{"row": 0}\n').hexdigest()}  questions.jsonl",
    ]
    (tmp_path / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff b")
    assert mod.write_manifest(tmp_path) != first


def test_only_allowlisted_image_formats_are_extracted() -> None:
    mod = _mod()
    assert mod.extension(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert mod.extension(b"\xff\xd8\xffrest") == "jpg"
    with pytest.raises(ValueError, match="allowlist"):
        mod.extension(b"<svg ...")
