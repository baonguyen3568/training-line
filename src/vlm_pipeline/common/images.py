"""Pinned project images (docker/IMAGES.lock): a job may only run an image whose local id matches its pinned row.

Single check point for every job (run_job): train, eval, serve, smoke, jobrun_cli, later plugin images (D16).
A new image that needs a probe gets a row first (status "PROBE, not approved"). There is deliberately no bypass flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.errors import VlmPipelineError


class ImageMismatchError(VlmPipelineError):
    """The local image tag no longer points at the image id pinned in docker/IMAGES.lock (tag rebuilt/replaced)."""

    code: ClassVar[str] = "SYS-000"  # TODO(fixcode): dedicated catalog code (image_digest_mismatch, D1/D16)


def pinned_image_id(lock_text: str, tag: str) -> str | None:
    """Local image id (column 5) of the IMAGES.lock row whose tag column starts with ``tag``."""
    for line in lock_text.splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 4 and cols[1].split(" ")[0] == tag and cols[4].startswith("sha256:"):
            return cols[4].split(" ")[0]
    return None


def check_pinned_image(root: Path, tag: str, actual_id: str | None) -> None:
    """Refuse to run a project image whose tag drifted from the reviewed, pinned id (bao-security, D1)."""
    expected = pinned_image_id((root / "docker" / "IMAGES.lock").read_text(encoding="utf-8"), tag)
    if expected is None or actual_id != expected:
        raise ImageMismatchError(
            "image tag does not match docker/IMAGES.lock",
            params={"image": tag, "expected": expected or "not pinned", "actual": actual_id or "missing"},
        )
