"""BUG-002: an exported checkpoint must contain every tensor of its base (Parse lost RADIO's norm_mean/norm_std)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from vlm_pipeline.train.ckpt_complete import (
    EXTRA_SHARD,
    INDEX_NAME,
    complete_from_base,
    missing_from_export,
    read_header,
    weight_map,
)

NORM_MEAN = "encoder.model_encoder.radio_model.input_conditioner.norm_mean"
NORM_STD = "encoder.model_encoder.radio_model.input_conditioner.norm_std"
DEAD = "decoder.embed_positions.weight"


def _shard(path: Path, tensors: dict[str, bytes], dtype: str = "F32") -> None:
    """Independent minimal safetensors writer (1-D tensors) so the code under test is not checked against itself."""
    header: dict[str, object] = {}
    offset = 0
    for name, data in tensors.items():
        header[name] = {"dtype": dtype, "shape": [len(data) // 4], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(tensors.values()))


def _tensor(path: Path, name: str) -> bytes:
    header, start = read_header(path)
    offsets = header[name]["data_offsets"]
    assert isinstance(offsets, list)
    begin, end = offsets
    return path.read_bytes()[start + begin : start + end]


def _f32(*xs: float) -> bytes:
    return struct.pack(f"<{len(xs)}f", *xs)


def _base(tmp_path: Path, *, tied: bool = False) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    _shard(
        base / "model.safetensors",
        {
            "encoder.conv1.weight": _f32(1, 2),
            NORM_MEAN: _f32(0.48, 0.45, 0.40),
            NORM_STD: _f32(0.26, 0.26, 0.27),
            DEAD: _f32(9, 9, 9, 9),
            "lm_head.weight": _f32(5, 5),
        },
    )
    (base / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}))
    return base


def _export(tmp_path: Path, *, tied: bool = False, extra: dict[str, bytes] | None = None) -> Path:
    out = tmp_path / "consolidated"
    out.mkdir()
    tensors = {"encoder.conv1.weight": _f32(7, 8), **(extra or {})}  # trained value differs from the base
    if not tied:
        tensors["lm_head.weight"] = _f32(6, 6)
    _shard(out / "model-00001-of-00001.safetensors", tensors)
    index = {
        "metadata": {"total_size": sum(map(len, tensors.values()))},
        "weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors"),
    }
    (out / INDEX_NAME).write_text(json.dumps(index))
    (out / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}))
    return out


def test_restores_missing_base_tensors_byte_for_byte(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    assert missing_from_export(base, out) == [DEAD, NORM_MEAN, NORM_STD]

    assert complete_from_base(base, out) == [DEAD, NORM_MEAN, NORM_STD]

    wm = weight_map(out)
    for name in (NORM_MEAN, NORM_STD, DEAD):
        assert wm[name] == EXTRA_SHARD
        assert _tensor(out / EXTRA_SHARD, name) == _tensor(base / "model.safetensors", name)
    assert read_header(out / EXTRA_SHARD)[0][NORM_MEAN]["shape"] == [3]
    index = json.loads((out / INDEX_NAME).read_text())
    assert index["metadata"]["vlm_restored_from_base"] == [DEAD, NORM_MEAN, NORM_STD]
    assert index["metadata"]["total_size"] == 4 * 4 + (3 + 3 + 4) * 4
    assert missing_from_export(base, out) == []


def test_trained_tensors_are_never_overwritten(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    complete_from_base(base, out)
    assert weight_map(out)["encoder.conv1.weight"] == "model-00001-of-00001.safetensors"
    assert _tensor(out / "model-00001-of-00001.safetensors", "encoder.conv1.weight") == _f32(7, 8)


def test_tied_head_alias_is_not_restored(tmp_path: Path) -> None:
    """With tie_word_embeddings the base lm_head is an alias: bringing it back would shadow the trained embedding."""
    base, out = _base(tmp_path, tied=True), _export(tmp_path, tied=True)
    restored = complete_from_base(base, out)
    assert "lm_head.weight" not in restored
    assert "lm_head.weight" not in weight_map(out)


def test_untied_head_missing_from_export_is_restored(tmp_path: Path) -> None:
    base = _base(tmp_path)
    out = _export(tmp_path, tied=True)
    (out / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
    assert "lm_head.weight" in complete_from_base(base, out)


def test_second_run_is_a_no_op(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    complete_from_base(base, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert complete_from_base(base, out) == []
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert {p.name for p in out.glob("*.safetensors")} == {EXTRA_SHARD, "model-00001-of-00001.safetensors"}


def test_later_run_keeps_tensors_restored_earlier(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    complete_from_base(base, out)
    _shard(base / "more.safetensors", {"decoder.new.bias": _f32(3)})
    index = {"metadata": {}, "weight_map": {**weight_map(base), "decoder.new.bias": "more.safetensors"}}
    (base / INDEX_NAME).write_text(json.dumps(index))

    assert complete_from_base(base, out) == ["decoder.new.bias"]
    for name in (NORM_MEAN, NORM_STD, DEAD, "decoder.new.bias"):
        assert weight_map(out)[name] == EXTRA_SHARD
    assert _tensor(out / EXTRA_SHARD, NORM_STD) == _f32(0.26, 0.26, 0.27)
    meta = json.loads((out / INDEX_NAME).read_text())["metadata"]
    assert meta["total_size"] == 4 * 4 + (3 + 3 + 4 + 1) * 4
    assert meta["vlm_restored_from_base"] == sorted([DEAD, NORM_MEAN, NORM_STD, "decoder.new.bias"])


def test_single_file_export_gets_an_index(tmp_path: Path) -> None:
    base = _base(tmp_path)
    out = tmp_path / "single"
    out.mkdir()
    _shard(out / "model.safetensors", {"encoder.conv1.weight": _f32(7, 8), "lm_head.weight": _f32(6, 6)})
    complete_from_base(base, out)
    wm = weight_map(out)
    assert wm["encoder.conv1.weight"] == "model.safetensors"
    assert wm[NORM_MEAN] == EXTRA_SHARD


def test_extra_shard_data_is_8_byte_aligned(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    complete_from_base(base, out)
    (size,) = struct.unpack("<Q", (out / EXTRA_SHARD).read_bytes()[:8])
    assert size % 8 == 0


def test_corrupt_base_offsets_are_rejected(tmp_path: Path) -> None:
    base, out = _base(tmp_path), _export(tmp_path)
    raw = (base / "model.safetensors").read_bytes()
    (base / "model.safetensors").write_bytes(raw[: len(raw) - 20])  # cuts into DEAD, a tensor to restore
    with pytest.raises(ValueError, match="data_offsets"):
        complete_from_base(base, out)
    assert not (out / EXTRA_SHARD).exists()
