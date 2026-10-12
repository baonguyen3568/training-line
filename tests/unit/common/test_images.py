"""The default training image is pinned and security-approved in docker/IMAGES.lock."""

from __future__ import annotations


def test_default_train_image_is_pinned_and_approved() -> None:
    from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE, is_approved, pinned_image_id
    from vlm_pipeline.common.paths import project_root

    text = (project_root() / "docker" / "IMAGES.lock").read_text(encoding="utf-8")
    assert pinned_image_id(text, DEFAULT_TRAIN_IMAGE) is not None
    assert is_approved(text, DEFAULT_TRAIN_IMAGE)
