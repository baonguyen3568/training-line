"""bao-security's image rules for the trial node (serve/trial.py open_image). Runs inside the image (Pillow), CPU."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
Image = pytest.importorskip("PIL.Image")

from vlm_pipeline.serve.trial import TrialInputError, open_image  # noqa: E402


def _save(path: Path, fmt: str, size: tuple[int, int] = (32, 16), **kw: object) -> Path:
    Image.new("RGB", size, "white").save(path, fmt, **kw)
    return path


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_allowed_formats_open_as_rgb(tmp_path: Path, fmt: str) -> None:
    img = open_image(_save(tmp_path / f"a.{fmt.lower()}", fmt))
    assert (img.mode, img.size) == ("RGB", (32, 16))


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_formats_are_refused_at_open(tmp_path: Path, fmt: str) -> None:
    path = _save(tmp_path / "a.png", fmt)  # extension lies: only the content decides, and it is not allowlisted
    with pytest.raises(TrialInputError) as ei:
        open_image(path)
    assert ei.value.reason == "unreadable"


def test_decompression_bomb_is_an_error_not_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vlm_pipeline.serve import trial

    path = _save(tmp_path / "big.png", "PNG", size=(200, 200))
    monkeypatch.setattr(trial, "MAX_IMAGE_PIXELS", 30_000)  # 40_000 px is over the limit but under 2x (a warning)
    with pytest.raises(TrialInputError) as ei:
        open_image(path)
    assert ei.value.reason == "too_many_pixels"


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    path = _save(tmp_path / "rot.jpg", "JPEG", exif=exif.tobytes())
    assert open_image(path).size == (16, 32)


def test_truncated_file_is_unreadable_and_symlink_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "a.png", "PNG")
    (tmp_path / "cut.png").write_bytes(path.read_bytes()[:40])
    with pytest.raises(TrialInputError) as ei:
        open_image(tmp_path / "cut.png")
    assert ei.value.reason == "unreadable"
    (tmp_path / "link.png").symlink_to(path)
    with pytest.raises(TrialInputError) as ei:
        open_image(tmp_path / "link.png")
    assert ei.value.reason == "not_a_file"
