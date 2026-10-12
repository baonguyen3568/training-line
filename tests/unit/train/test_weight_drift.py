"""weight_drift: moved = differs from base beyond bf16 rounding (needs numpy -> skipped on a bare host)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from vlm_pipeline.train import weight_drift  # noqa: E402


def _save(path: Path, tensors: dict[str, Any]) -> None:
    header, blobs, offset = {}, [], 0
    for key, arr in tensors.items():
        raw = arr.tobytes()
        dtype = {"float32": "F32", "int64": "I64", "uint16": "BF16"}[str(arr.dtype)]  # uint16 = raw bf16 bits
        header[key] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def _bf16(x: Any) -> Any:
    return weight_drift._round_bf16(x.view(np.uint32)).view(np.float32)


def test_bf16_rounding_is_not_movement_and_updates_are(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, 1000).astype(np.float32)
    trained = _bf16(w).copy()
    trained[:100] += 0.01  # 10% really updated
    _save(tmp_path / "base", {"encoder.blocks.0.w": w, "decoder.layers.0.w": w, "decoder.idx": np.arange(3)})
    _save(
        tmp_path / "ckpt", {"encoder.blocks.0.w": _bf16(w), "decoder.layers.0.w": trained, "decoder.idx": np.arange(3)}
    )

    groups = weight_drift.drift(tmp_path / "base", tmp_path / "ckpt")

    assert groups["encoder.blocks"].moved == 0
    assert groups["encoder.blocks"].changed_bits > 0  # rounded, not trained
    assert groups["decoder.layers"].moved == 100
    assert groups["decoder.idx"].moved == 0
    assert weight_drift.check(groups, frozen=["encoder"], trainable=["decoder.layers"], min_moved_frac=0.05) == []
    problems = weight_drift.check(groups, frozen=["decoder.layers"], trainable=["decoder.layers"], min_moved_frac=0.5)
    assert len(problems) == 2


def test_key_mismatch_is_an_error(tmp_path: Path) -> None:
    _save(tmp_path / "base", {"a.w": np.zeros(2, np.float32)})
    _save(tmp_path / "ckpt", {"b.w": np.zeros(2, np.float32)})
    with pytest.raises(ValueError, match="key sets differ"):
        weight_drift.drift(tmp_path / "base", tmp_path / "ckpt")


@pytest.mark.parametrize(
    ("key", "group"),
    [
        ("decoder.layers.3.fc1.weight", "decoder.layers"),
        ("encoder.model_encoder.x.y", "encoder.model_encoder"),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_default_group(key: str, group: str) -> None:
    assert weight_drift.default_group(key) == group


def _groups(tmp_path: Path) -> dict[str, weight_drift.GroupDrift]:
    w = np.ones(4, np.float32)
    _save(tmp_path / "base", {"encoder.conv1.w": w, "encoder_extra.x.w": w, "decoder.layers.0.w": w})
    _save(tmp_path / "ckpt", {"encoder.conv1.w": w, "encoder_extra.x.w": w + 1, "decoder.layers.0.w": w + 1})
    return weight_drift.drift(tmp_path / "base", tmp_path / "ckpt")


def test_unmatched_prefix_is_a_problem_not_a_vacuous_pass(tmp_path: Path) -> None:
    problems = weight_drift.check(_groups(tmp_path), frozen=["encodr"], trainable=["decoder.layer_s"], min_moved_frac=0)
    assert problems == [
        "--frozen prefix 'encodr' matches no group",
        "--trainable prefix 'decoder.layer_s' matches no group",
    ]


def test_prefix_matches_on_dot_boundaries(tmp_path: Path) -> None:
    # "encoder" must not catch "encoder_extra.x" (which moved)
    assert weight_drift.check(_groups(tmp_path), frozen=["encoder"], trainable=["decoder"], min_moved_frac=1.0) == []


def test_min_moved_frac_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_moved_frac"):
        weight_drift.check(_groups(tmp_path), frozen=[], trainable=[], min_moved_frac=10)


def test_bf16_checkpoint_of_fp32_base(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, 64).astype(np.float32)
    bf16_bits = (weight_drift._round_bf16(w.view(np.uint32)) >> 16).astype(np.uint16)
    trained = bf16_bits.copy()
    trained[:8] += 1  # one bf16 ulp on 8 elements
    _save(tmp_path / "base", {"a.b.w": w, "c.d.w": w})
    _save(tmp_path / "ckpt", {"a.b.w": bf16_bits, "c.d.w": trained})
    groups = weight_drift.drift(tmp_path / "base", tmp_path / "ckpt")
    assert groups["a.b"].moved == 0
    assert groups["c.d"].moved == 8


def test_shape_mismatch_raises_for_integer_tensors_too(tmp_path: Path) -> None:
    _save(tmp_path / "base", {"a.idx": np.arange(3)})
    _save(tmp_path / "ckpt", {"a.idx": np.arange(4)})
    with pytest.raises(ValueError, match=r"a\.idx"):
        weight_drift.drift(tmp_path / "base", tmp_path / "ckpt")
