"""Image rules shared by every place that accepts a user image (bao-security): uploads, trial node, local datasets.

Format allowlist at open (Pillow plugins outside it are never used: GD/EPS/... advisories), a decompression bomb is an
error, at most ``MAX_IMAGE_BYTES`` per file and ``MAX_IMAGE_PIXELS`` per image. Pillow is imported lazily.
"""

from __future__ import annotations

import io
import stat
import warnings
from pathlib import Path
from typing import IO, Any

IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


class ImageRuleError(ValueError):
    """``reason``: missing | not_a_file | too_large | too_many_pixels | unreadable."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def decode_image(fp: str | Path | IO[bytes]) -> Any:
    """Fully decoded RGB image under the rules (first frame only, EXIF orientation applied), or ``ImageRuleError``."""
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            img: Any = Image.open(fp, formats=list(IMAGE_FORMATS))
            img.load()  # first frame only for animated images: no seek
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ImageRuleError("too_many_pixels") from None
        except (OSError, ValueError, SyntaxError):  # UnidentifiedImageError is an OSError
            raise ImageRuleError("unreadable") from None


def _check_file(path: Path) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        raise ImageRuleError("missing") from None
    except OSError:
        raise ImageRuleError("not_a_file") from None
    if not stat.S_ISREG(st.st_mode):  # a symlink is never followed
        raise ImageRuleError("not_a_file")
    if st.st_size > MAX_IMAGE_BYTES:
        raise ImageRuleError("too_large")


def open_image(path: Path) -> Any:
    """RGB image from a file: regular file (no symlink), size limit, then ``decode_image``."""
    _check_file(path)
    return decode_image(path)


def open_image_bytes(data: bytes) -> Any:
    """RGB image from bytes (e.g. benchmark parquet cells): size limit, then ``decode_image``."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageRuleError("too_large")
    return decode_image(io.BytesIO(data))


def probe_image(path: Path) -> tuple[int, int, str]:
    """(width, height, format) from the header only (no full decode): fast checks over a whole dataset folder.

    Same rules as a full open: regular file (no symlink), size limit, format allowlist, pixel limit. A truncated
    image body passes the header probe and fails only when the job decodes it (DATA-002 at train time).
    """
    from PIL import Image

    _check_file(path)
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(path, formats=list(IMAGE_FORMATS)) as img:
                width, height, fmt = img.width, img.height, str(img.format)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ImageRuleError("too_many_pixels") from None
        except (OSError, ValueError, SyntaxError):
            raise ImageRuleError("unreadable") from None
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageRuleError("too_many_pixels")
    return width, height, fmt


__all__ = [
    "IMAGE_FORMATS",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_PIXELS",
    "ImageRuleError",
    "decode_image",
    "open_image",
    "open_image_bytes",
    "probe_image",
]
