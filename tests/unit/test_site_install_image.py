"""/docs/cai-dat names the image the app really trains with (common.images.DEFAULT_TRAIN_IMAGE), and only that one."""

from __future__ import annotations

import re

from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.paths import project_root

PAGE = project_root() / "site" / "src" / "content" / "docs" / "docs" / "cai-dat.mdx"


def test_the_install_page_follows_the_default_train_image() -> None:
    text = PAGE.read_text(encoding="utf-8")
    tag = DEFAULT_TRAIN_IMAGE.rsplit(":", 1)[1]
    hint = f"image bump -> update /docs/cai-dat ({PAGE.relative_to(project_root())}) to {DEFAULT_TRAIN_IMAGE}"
    assert DEFAULT_TRAIN_IMAGE in text, hint
    assert f"VLM_TAG={tag}" in text, hint
    # Another tag on the page is stale. If the page ever names a rollback image, allow exactly a ROLLBACK_TRAIN_IMAGE
    # constant from common.images here; never loosen the pattern (bao-cleanup).
    others = sorted(set(re.findall(r"vlm-automodel:[\w.-]+", text)) - {DEFAULT_TRAIN_IMAGE})
    stale = sorted(set(re.findall(r"VLM_TAG=([\w.-]+)", text)) - {tag})
    assert (others, stale) == ([], []), hint
