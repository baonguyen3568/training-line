"""The Dataset artifact descriptor shared by data, train and eval workers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from vlm_pipeline.common.dataset_ref import DatasetRef, read_dataset_ref, write_dataset_ref

REV = "7f0115a4b758a71d6473b8d085751692da2fef98"


def test_descriptor_written_before_splits_reads_as_train(tmp_path: Path) -> None:
    old = tmp_path / "dataset.json"
    old.write_text(
        json.dumps({"repo": "naver-clova-ix/cord-v2", "revision": REV, "path": "/workspace/d", "limit": 100})
    )
    ref = read_dataset_ref(old)
    assert (ref.split, ref.limit) == ("train", 100)
    assert ref.hash == f"naver-clova-ix/cord-v2@{REV}#limit=100"  # pre-split cache keys stay valid


def test_hash_keeps_the_revision_readable_and_round_trips(tmp_path: Path) -> None:
    ref = DatasetRef(repo="naver-clova-ix/cord-v2", revision=REV, path="/workspace/d", split="validation", limit=10)
    assert re.search(r"@([0-9a-f]{40})(?:#|$)", ref.hash)  # RunManifest's dataset_revision rule
    write_dataset_ref(tmp_path / "v.json", ref)
    assert read_dataset_ref(tmp_path / "v.json") == ref


def test_local_data_hash_uses_the_path() -> None:
    assert DatasetRef(path="/workspace/data/mine").hash == "/workspace/data/mine"
