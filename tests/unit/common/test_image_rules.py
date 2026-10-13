"""One set of image rules for uploads, trial node, local datasets and benchmark bytes (bao-security)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from vlm_pipeline.common import image_rules as ir

Image = pytest.importorskip("PIL.Image")


def _bytes(fmt: str = "PNG", size: tuple[int, int] = (8, 6)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format=fmt)
    return buf.getvalue()


def test_bytes_and_file_decode_to_rgb(tmp_path: Path) -> None:
    assert ir.open_image_bytes(_bytes("WEBP")).mode == "RGB"
    (tmp_path / "a.jpg").write_bytes(_bytes("JPEG"))
    assert ir.open_image(tmp_path / "a.jpg").size == (8, 6)
    assert ir.probe_image(tmp_path / "a.jpg") == (8, 6, "JPEG")


@pytest.mark.parametrize(("payload", "reason"), [(_bytes("GIF"), "unreadable"), (b"%!PS-Adobe-3.0 EPSF", "unreadable")])
def test_formats_outside_the_allowlist_are_unreadable(payload: bytes, reason: str) -> None:
    with pytest.raises(ir.ImageRuleError) as ei:
        ir.open_image_bytes(payload)
    assert ei.value.reason == reason


def test_size_symlink_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ir, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ir.ImageRuleError, match="too_large"):
        ir.open_image_bytes(b"x" * 11)
    (tmp_path / "real.png").write_bytes(b"x" * 5)
    (tmp_path / "link.png").symlink_to(tmp_path / "real.png")
    for name, reason in (("link.png", "not_a_file"), ("nope.png", "missing")):
        with pytest.raises(ir.ImageRuleError) as ei:
            ir.probe_image(tmp_path / name)
        assert ei.value.reason == reason
