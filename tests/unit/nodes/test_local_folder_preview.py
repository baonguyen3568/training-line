"""DATA-V: the canvas warns (DATA-006) about a local folder the run's scan would refuse, from image headers only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.data import local_folder
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY

pil = pytest.importorskip("PIL.Image")


def _wf(path: str) -> dict[str, Any]:
    node = {"id": "f", "type": "data.local_folder", "version": "1.0.0", "params": {"path": path}}
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": [node], "edges": []}


def _folder(root: Path, lines: list[dict[str, Any]], images: tuple[str, ...] = ("a.png",)) -> Path:
    folder = root / "data" / "mine"
    folder.mkdir(parents=True, exist_ok=True)
    for name in images:
        pil.new("RGB", (32, 24)).save(folder / name)
    (folder / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return folder


def _data006(root: Path) -> list[dict[str, object]]:
    return [dict(i.params) | {"severity": i.severity} for i in validate(_wf("data/mine"), REGISTRY, workspace=root)
            if i.code == "DATA-006"]  # fmt: skip


def test_a_clean_folder_gets_no_warning(tmp_path: Path) -> None:
    _folder(tmp_path, [{"image": "a.png", "target": "x"}])
    assert _data006(tmp_path) == []


def test_a_missing_image_is_warned_with_its_line(tmp_path: Path) -> None:
    _folder(tmp_path, [{"image": "a.png", "target": "x"}, {"image": "gone.png", "target": "y"}])
    [warn] = _data006(tmp_path)
    assert warn["severity"] == "warn"  # a hint: the run's own scan is what refuses
    assert (warn["first_line"], warn["problem_count"], warn["node"]) == (2, 1, "f")
    assert str(warn["first_reason"]).startswith("image_")


def test_a_missing_folder_is_warned(tmp_path: Path) -> None:
    [warn] = _data006(tmp_path)
    assert warn["first_reason"] == "labels_missing"


def test_the_preview_never_hashes_and_is_cached_until_the_folder_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _folder(tmp_path, [{"image": "a.png", "target": "x"}])
    monkeypatch.setattr(local_folder, "_sha256", lambda _p: pytest.fail("the preview reads headers only"))
    first = local_folder.preview_folder(folder)
    assert first is not None
    assert first is local_folder.preview_folder(folder)  # cached
    (folder / "labels.jsonl").write_text(json.dumps({"image": "nope.png", "target": "x"}) + "\n")
    again = local_folder.preview_folder(folder)
    assert again is not None
    assert again.problem_count == 1


def test_a_folder_over_the_time_budget_gets_no_verdict(tmp_path: Path) -> None:
    folder = _folder(tmp_path, [{"image": "a.png", "target": "x"}] * 3)
    calls: list[int] = []
    ticks = iter(range(100))

    def clock() -> int:
        calls.append(1)
        return next(ticks)

    assert local_folder.preview_folder(folder, budget_s=0.5, clock=clock) is None
    used = len(calls)
    assert local_folder.preview_folder(folder, budget_s=0.5, clock=clock) is None  # cached "no verdict"
    assert len(calls) == used  # the folder was not read again


def test_a_preview_never_yields_records(tmp_path: Path) -> None:
    folder = _folder(tmp_path, [{"image": "a.png", "target": "x"}])
    preview = local_folder.preview_folder(folder)
    assert preview is not None
    assert preview.records == []
