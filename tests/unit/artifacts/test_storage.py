"""Unit tests for ``vlm_pipeline.artifacts.storage`` (no real docker, no real host paths)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from vlm_pipeline.artifacts.storage import (
    dedup_layers_bytes,
    docker_project_gb,
    du_gb,
    pinned_images,
    project_image_ids,
)
from vlm_pipeline.common.paths import GIB


def test_du_counts_hardlinks_once_and_ignores_symlink_targets(tmp_path: Path) -> None:
    data = tmp_path / "d"
    data.mkdir()
    blob = data / "blob"
    blob.write_bytes(b"x" * 1_000_000)
    os.link(blob, data / "hardlink")
    outside = tmp_path / "outside"
    outside.write_bytes(b"y" * 5_000_000)
    (data / "sym").symlink_to(outside)

    size = du_gb(data) * GIB
    assert size >= 1_000_000
    assert size < 2_000_000  # hard link once, symlink target not followed


def test_du_missing_path_is_zero(tmp_path: Path) -> None:
    assert du_gb(tmp_path / "nope") == 0.0


def test_pinned_images_parses_lock_columns(tmp_path: Path) -> None:
    lock = tmp_path / "IMAGES.lock"
    lock.write_text("# header | ignored\nautomodel | nvcr.io/nvidia/nemo-automodel:26.08 | sha256:abc | x\n\n")
    assert pinned_images(lock) == {"nvcr.io/nvidia/nemo-automodel:26.08"}
    assert pinned_images(tmp_path / "missing") == set()


def test_dedup_layers_counts_shared_base_once() -> None:
    base = (["l1", "l2"], 30 * GIB)
    derived = (["l1", "l2", "l3"], 31 * GIB)
    unrelated = (["z1"], 5 * GIB)
    assert dedup_layers_bytes([derived, base, unrelated]) == 36 * GIB


def _fake_runner(outputs: dict[tuple[str, ...], str | None]):  # type: ignore[no-untyped-def]
    def run(argv: Sequence[str]) -> str | None:
        return outputs.get(tuple(argv), "")

    return run


LISTING = ("images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}")
LABELLED = ("images", "--quiet", "--filter", "label=project=vlm-training")


def test_project_image_ids_selects_only_ours() -> None:
    run = _fake_runner(
        {
            LISTING: "vlm-automodel:26.08-1\tid1\n"
            "someone/vlm-x:1\tid2\n"
            "nvcr.io/nvidia/nemo-automodel:26.08\tid3\n"
            "comfy-ui:1.0\tid4\n"
            "labelled-tool:1\tid5\n",
            LABELLED: "id5\n",
        }
    )
    assert project_image_ids({"nvcr.io/nvidia/nemo-automodel:26.08"}, run) == {"id1", "id3", "id5"}


def test_docker_unavailable_returns_none() -> None:
    run = _fake_runner({LISTING: None, LABELLED: None})
    assert project_image_ids(set(), run) is None
    assert docker_project_gb(set(), run) is None


@pytest.mark.parametrize("inspect_out", ["", "garbage", '["l1"]\tnot-a-number'])
def test_docker_project_gb_skips_unparseable_inspect(inspect_out: str) -> None:
    inspect = ("image", "inspect", "--format", "{{json .RootFS.Layers}}\t{{.Size}}", "id1")
    run = _fake_runner({LISTING: "vlm-a:1\tid1\n", LABELLED: "", inspect: inspect_out})
    assert docker_project_gb(set(), run) == 0.0
