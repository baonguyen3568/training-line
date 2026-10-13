"""ADR-012 B3: remote code and machine paths are judged on the export folder's real content."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_pipeline.export import portable


def _cfg(folder: Path, name: str, data: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data))


def test_native_folder_has_no_findings(tmp_path: Path) -> None:
    _cfg(tmp_path, "config.json", {"model_type": "qwen3_vl", "vision_config": {"depth": 2}})
    _cfg(tmp_path, "generation_config.json", {"do_sample": False})
    assert portable.remote_code_findings(tmp_path) == []


def test_auto_map_in_a_sub_config_is_found_even_if_the_catalog_says_native(tmp_path: Path) -> None:
    _cfg(tmp_path, "config.json", {"model_type": "x", "vision_config": {"auto_map": {"AutoConfig": "m.C"}}})
    assert portable.remote_code_findings(tmp_path) == ["config.json: vision_config has auto_map"]


def test_python_next_to_weights_and_unreadable_config_are_findings(tmp_path: Path) -> None:
    _cfg(tmp_path, "config.json", {"auto_map": {"AutoModel": "m.M"}})
    (tmp_path / "modeling_x.py").write_text("print('hi')\n")
    (tmp_path / "preprocessor_config.json").write_text("{not json")
    findings = portable.remote_code_findings(tmp_path)
    assert findings[0] == "modeling_x.py: Python file next to the weights"
    assert "config.json: <root> has auto_map" in findings
    assert any(f.startswith("preprocessor_config.json: unreadable") for f in findings)


def test_machine_paths_are_found_and_name_or_path_is_neutralized(tmp_path: Path) -> None:
    _cfg(
        tmp_path, "config.json", {"_name_or_path": "/workspace/.cache/hf/x", "text_config": {"_name_or_path": "/opt/m"}}
    )
    assert portable.absolute_path_findings(tmp_path) == [
        "config.json: contains a machine path (/workspace)",
        "config.json: _name_or_path is a machine path",
        "config.json: text_config._name_or_path is a machine path",
    ]
    changed = portable.neutralize_name_or_path(tmp_path, "Qwen/Qwen3.5-2B@15852e8c")
    assert changed == ["config.json: _name_or_path", "config.json: text_config._name_or_path"]
    assert portable.absolute_path_findings(tmp_path) == []
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["text_config"]["_name_or_path"] == "Qwen/Qwen3.5-2B@15852e8c"


def test_auto_map_in_any_json_is_remote_code(tmp_path: Path) -> None:
    _cfg(tmp_path, "extra.json", {"custom_pipelines": {"x": {"impl": "p.P"}}, "sub": {"auto_map": {"A": "m.C"}}})
    assert portable.remote_code_findings(tmp_path) == [
        "extra.json: <root> has custom_pipelines",
        "extra.json: sub has auto_map",
    ]


def test_machine_paths_in_any_text_file_are_found(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("trained by /home/someone/project\n")
    (tmp_path / "trainer_state.json").write_text(json.dumps({"best_model": "/workspace/outputs/x"}))
    assert portable.absolute_path_findings(tmp_path) == [
        "notes.txt: contains a machine path (/home/)",
        "trainer_state.json: contains a machine path (/workspace)",
    ]
