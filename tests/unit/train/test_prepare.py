from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vlm_pipeline.train.prepare import Bridge, PinnedCode, PrepareError, prepare_bridged_model, verify_code


def _snapshot(root: Path, name: str, files: dict[str, str], config: dict[str, object] | None = None) -> PinnedCode:
    snap = root / "hub" / name
    snap.mkdir(parents=True)
    lines = []
    for fname, body in files.items():
        (snap / fname).write_text(body)
        if fname.endswith(".py"):
            lines.append(f"{hashlib.sha256(body.encode()).hexdigest()}  {fname}")
    if config is not None:
        (snap / "config.json").write_text(json.dumps(config))
    sha = root / f"{name}.sha256"
    sha.write_text("\n".join(lines) + "\n")
    return PinnedCode(f"org/{name}", "a" * 40, snap, sha)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[PinnedCode, PinnedCode]:
    radio = _snapshot(tmp_path, "radio", {"hf_model.py": "X=1\n"}, {"arch": "RADIO"})
    model = _snapshot(
        tmp_path,
        "parse",
        {"modeling.py": "Y=2\n", "model.safetensors": "weights"},
        {
            "auto_map": {"AutoConfig": "modeling.Cfg"},
            "encoder": {
                "_name_or_path": "org/radio",
                "auto_map": {"AutoConfig": "org/radio--hf_model.RADIOConfig", "AutoModel": "org/radio--hf_model.RM"},
            },
        },
    )
    return model, radio


def test_prepare_rewrites_bridges_and_copies_vetted_code(tmp_path: Path, setup: tuple[PinnedCode, PinnedCode]) -> None:
    model, radio = setup
    out = tmp_path / "prepared" / "parse_aaaa"
    model_dir = prepare_bridged_model(model, [Bridge(radio, "radio_aaaa")], out, project_root=tmp_path)
    cfg = json.loads((model_dir / "config.json").read_text())
    enc = cfg["encoder"]
    assert enc["_name_or_path"] == "/workspace/prepared/parse_aaaa/radio_aaaa"
    assert enc["auto_map"]["AutoConfig"] == "/workspace/prepared/parse_aaaa/radio_aaaa--hf_model.RADIOConfig"
    assert cfg["auto_map"] == {"AutoConfig": "modeling.Cfg"}  # same-repo references untouched
    assert (out / "radio_aaaa" / "hf_model.py").read_text() == "X=1\n"
    assert not (out / "radio_aaaa" / "hf_model.py").is_symlink()
    assert (model_dir / "model.safetensors").is_symlink()
    assert (model_dir / "modeling.py").read_text() == "Y=2\n"


def test_tampered_code_is_rejected(tmp_path: Path, setup: tuple[PinnedCode, PinnedCode]) -> None:
    model, radio = setup
    (radio.snapshot_dir / "hf_model.py").write_text("import os; os.system('x')\n")
    assert verify_code(radio)
    with pytest.raises(PrepareError, match="sha256 mismatch"):
        prepare_bridged_model(model, [Bridge(radio, "radio_aaaa")], tmp_path / "p" / "x", project_root=tmp_path)


def test_unvetted_python_file_is_rejected(tmp_path: Path, setup: tuple[PinnedCode, PinnedCode]) -> None:
    model, radio = setup
    (radio.snapshot_dir / "extra.py").write_text("pass\n")
    with pytest.raises(PrepareError, match="unvetted"):
        prepare_bridged_model(model, [Bridge(radio, "radio_aaaa")], tmp_path / "p" / "x", project_root=tmp_path)


def test_double_dash_paths_are_refused(tmp_path: Path, setup: tuple[PinnedCode, PinnedCode]) -> None:
    model, radio = setup
    with pytest.raises(PrepareError, match="--"):
        prepare_bridged_model(model, [Bridge(radio, "r")], tmp_path / "models--x", project_root=tmp_path)


def test_unknown_cross_repo_reference_left_is_refused(tmp_path: Path, setup: tuple[PinnedCode, PinnedCode]) -> None:
    model, radio = setup
    cfg = json.loads((model.snapshot_dir / "config.json").read_text())
    cfg["vision"] = {"auto_map": {"AutoModel": "evil/repo--m.C"}}
    (model.snapshot_dir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(PrepareError, match="unpinned cross-repo"):
        prepare_bridged_model(model, [Bridge(radio, "radio_aaaa")], tmp_path / "p" / "x", project_root=tmp_path)
