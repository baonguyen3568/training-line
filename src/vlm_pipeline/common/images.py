"""Pinned project images (docker/IMAGES.lock): a job may only run an image whose local id matches its pinned row.

Single check point for every job (run_job): train, eval, serve, smoke, jobrun_cli, later plugin images (D16).
A new image that needs a probe gets a row first (status "PROBE, not approved"). There is deliberately no bypass flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.approval import is_security_approved
from vlm_pipeline.common.errors import VlmPipelineError

# The one place that names the default training image (app worker, smoke CLI, container gates).
DEFAULT_TRAIN_IMAGE = "vlm-automodel:26.08-7"


# Extra groups a job may join, PER IMAGE ID (security 2026-10-14): never by tag, since anyone can retag another image
# locally. vlm-nemo-rl:0.7.0-1 (docker/IMAGES.lock, PROBE ONLY) keeps /opt/nemo-rl 660/770 root:root, so its
# non-root user needs group 0. Never offered to any other image (a job in group 0 can read root-group files).
GROUP_ADD_ALLOWED: dict[str, frozenset[int]] = {
    "sha256:4c740d252ef90651621224005623546b4931582fa2eeb9ce816ba9d7576a4add": frozenset({0}),  # vlm-nemo-rl:0.7.0-1
}


def group_add_problem(image: str, groups: tuple[int, ...]) -> str | None:
    """Why ``image`` may not run with ``groups`` added, or None (no groups is always fine).

    ``image`` must be the RESOLVED image id run_job runs by (the launcher pins ``job.image`` to it), never a tag.
    """
    if not groups:
        return None
    allowed = GROUP_ADD_ALLOWED.get(image, frozenset())
    extra = sorted(set(groups) - allowed)
    return f"image {image[:19]} may not join group(s) {extra} (allowed by image id only)" if extra else None


class ImageMismatchError(VlmPipelineError):
    """The local image tag no longer points at the image id pinned in docker/IMAGES.lock (tag rebuilt/replaced)."""

    code: ClassVar[str] = "SEC-004"  # IMAGE_DIGEST_MISMATCH


def _row(lock_text: str, tag: str) -> list[str] | None:
    for line in lock_text.splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 4 and cols[1].split(" ")[0] == tag and cols[4].startswith("sha256:"):
            return cols
    return None


def pinned_image_id(lock_text: str, tag: str) -> str | None:
    """Local image id (column 5) of the IMAGES.lock row whose tag column starts with ``tag``."""
    row = _row(lock_text, tag)
    return row[4].split(" ")[0] if row else None


def is_approved(lock_text: str, tag: str) -> bool:
    """Review column (6) must START with exactly ``bao-security APPROVED`` (our own file: no free-text parsing)."""
    row = _row(lock_text, tag)
    return row is not None and len(row) > 6 and is_security_approved(row[6])


def check_pinned_image(root: Path, tag: str, actual_id: str | None, *, require_approved: bool = False) -> None:
    """Refuse to run a project image whose tag drifted from the reviewed, pinned id (bao-security, D1).

    ``require_approved`` (jobs started from the app for users): a PROBE row is not enough.
    """
    lock_text = (root / "docker" / "IMAGES.lock").read_text(encoding="utf-8")
    expected = pinned_image_id(lock_text, tag)
    if expected is None or actual_id != expected or (require_approved and not is_approved(lock_text, tag)):
        raise ImageMismatchError(
            "image tag does not match docker/IMAGES.lock",
            params={"image": tag, "expected": expected or "not pinned", "actual": actual_id or "missing"},
        )
