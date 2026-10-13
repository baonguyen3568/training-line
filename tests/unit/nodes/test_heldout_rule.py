"""DATA-007: a local folder under data/eval/ (extracted benchmark test data) never reaches a fine-tune node's data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.heldout_data import is_heldout_folder
from vlm_pipeline.graph import validate
from vlm_pipeline.nodes import REGISTRY

QWEN = "qwen3.5-0.8b"


def _n(node_id: str, type_id: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "type": type_id, "version": "1.0.0", "params": params}


def _e(src: str, out: str, dst: str, inp: str) -> dict[str, Any]:
    return {"from": [src, out], "to": [dst, inp]}


def _wf(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [_n("m", "model.hf_pick", model=QWEN), _n("t", "train.lora_vlm"), *nodes]
    edges = [_e("m", "model", "t", "base_model"), *edges]
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": nodes, "edges": edges}


def _folder_to_train(path: str) -> dict[str, Any]:
    return _wf([_n("f", "data.local_folder", path=path)], [_e("f", "dataset", "t", "train_data")])


def _data007(workflow: dict[str, Any], workspace: Path | None) -> list[tuple[str | None, dict[str, object]]]:
    issues = validate(workflow, REGISTRY, workspace=workspace)
    return [(i.node_id, dict(i.params)) for i in issues if i.code == "DATA-007"]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "data" / "eval" / "ocrbench-92a54bd1").mkdir(parents=True)
    (tmp_path / "data" / "train").mkdir(parents=True)
    return tmp_path


def test_direct_path_under_data_eval_is_blocked(ws: Path) -> None:
    assert _data007(_folder_to_train("data/eval/ocrbench-92a54bd1"), ws) == [("f", {"node": "f", "target_node": "t"})]
    assert _data007(_folder_to_train("data/eval"), ws) == [("f", {"node": "f", "target_node": "t"})]


def test_symlink_into_data_eval_is_blocked(ws: Path) -> None:
    (ws / "data" / "mine").symlink_to(ws / "data" / "eval" / "ocrbench-92a54bd1")
    assert _data007(_folder_to_train("data/mine"), ws) == [("f", {"node": "f", "target_node": "t"})]


def test_dotdot_out_of_data_eval_is_sec001_not_a_heldout_false_positive(ws: Path) -> None:
    issues = validate(_folder_to_train("data/eval/../train"), REGISTRY, workspace=ws)
    assert ("SEC-001", "f") in [(i.code, i.node_id) for i in issues]  # '..' is never accepted in a local path
    assert not [i for i in issues if i.code == "DATA-007"]


def test_other_folders_and_lookalikes_pass(ws: Path) -> None:
    assert _data007(_folder_to_train("data/train"), ws) == []
    assert _data007(_folder_to_train("data/evaluation"), ws) == []  # a name prefix is not the folder


def test_traced_through_split_filter_dedupe_and_mix_into_val_data(ws: Path) -> None:
    nodes = [
        _n("a", "data.local_folder", path="data/train"),
        _n("f", "data.local_folder", path="data/eval/ocrbench-92a54bd1"),
        _n("fl", "data.filter"),
        _n("dd", "data.dedupe"),
        _n("mx", "data.mix"),
        _n("sp", "data.split"),
    ]
    edges = [
        _e("f", "dataset", "fl", "dataset"),
        _e("fl", "dataset", "dd", "dataset"),
        _e("a", "dataset", "mx", "dataset_1"),
        _e("dd", "dataset", "mx", "dataset_2"),
        _e("mx", "dataset", "sp", "dataset"),
        _e("sp", "train", "t", "train_data"),
        _e("sp", "validation", "t", "val_data"),
    ]
    assert _data007(_wf(nodes, edges), ws) == [("f", {"node": "f", "target_node": "t"})]  # once, not per port


def test_dedupe_heldout_port_is_where_test_data_belongs(ws: Path) -> None:
    nodes = [
        _n("a", "data.local_folder", path="data/train"),
        _n("f", "data.local_folder", path="data/eval/ocrbench-92a54bd1"),
        _n("dd", "data.dedupe"),
    ]
    edges = [
        _e("a", "dataset", "dd", "dataset"),
        _e("f", "dataset", "dd", "heldout"),
        _e("dd", "dataset", "t", "train_data"),
    ]
    assert _data007(_wf(nodes, edges), ws) == []


def test_quantize_calibration_on_data_eval_is_blocked(ws: Path) -> None:
    """Calibrating FP8/NVFP4 on the test images and then scoring on them flatters the gate (tester, R3)."""
    nodes = [_n("f", "data.local_folder", path="data/eval/ocrbench-92a54bd1"), _n("q", "quantize.modelopt_ptq")]
    edges = [_e("m", "model", "q", "model"), _e("f", "dataset", "q", "calib_data")]
    assert _data007(_wf(nodes, edges), ws) == [("f", {"node": "f", "target_node": "q"})]


def test_without_a_workspace_the_path_text_decides() -> None:
    assert _data007(_folder_to_train("data/eval/x"), None) == [("f", {"node": "f", "target_node": "t"})]
    assert _data007(_folder_to_train("data/train"), None) == []


def test_is_heldout_folder_accepts_absolute_host_paths(ws: Path) -> None:
    assert is_heldout_folder(ws, ws / "data" / "eval" / "ocrbench-92a54bd1")
    assert not is_heldout_folder(ws, ws / "data" / "train")


@pytest.mark.parametrize("locale", ["vi", "en"])
def test_issue_renders_in_both_locales(ws: Path, locale: str) -> None:
    from dataclasses import asdict

    from vlm_pipeline.app.main import _render_issue

    [issue] = [i for i in validate(_folder_to_train("data/eval/x"), REGISTRY, workspace=ws) if i.code == "DATA-007"]
    out = _render_issue(asdict(issue), locale)
    assert out["title"]
    assert "{" not in out["title"] + out["what"]
    assert "data/eval" in out["what"]
