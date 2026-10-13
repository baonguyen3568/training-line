"""CORD chat worker helpers: a folder run reads the PNG the vLLM batch reads, and records its sha (parity gate)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "eval_cord_chat_worker.py"


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("eval_cord_chat_worker", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_folder_rows_give_the_png_its_sha_and_the_target(tmp_path: Path) -> None:
    mod = _mod()
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 10, 10)).save(buf, format="PNG")
    (tmp_path / "test_0000.png").write_bytes(buf.getvalue())
    target = {"total": {"total_price": "1.000"}}
    (tmp_path / "labels.jsonl").write_text(
        json.dumps({"image": "test_0000.png", "target": target, "prompt": "p"}) + "\n"
    )
    rows = mod.folder_items(tmp_path)
    image, gt, sha = mod.load_item(0, None, tmp_path, rows)
    assert image.size == (32, 24)
    assert image.mode == "RGB"
    assert gt == target
    assert sha == hashlib.sha256(buf.getvalue()).hexdigest()


def test_a_label_pointing_outside_the_folder_or_past_its_end_stops_the_run(tmp_path: Path) -> None:
    import pytest

    mod = _mod()
    rows = [{"image": "../secret.png", "target": {}}]
    with pytest.raises(SystemExit, match="not allowed"):
        mod.load_item(0, None, tmp_path, rows)
    with pytest.raises(SystemExit, match="1 rows"):
        mod.load_item(1, None, tmp_path, rows)
