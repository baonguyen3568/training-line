"""Post-train weight drift: how much of each parameter group actually moved away from the base model.

Catches recipes that "train" without changing the weights, e.g. pure-bf16 params where AdamW updates smaller than
half a bf16 ulp are rounded away (P-3: 5.3% of decoder elements moved), and frozen groups that were not frozen.
An element counts as moved when it differs from the base value AND from the base value rounded to bf16 (a bf16
training run that saves fp32 rounds untouched weights, which is not training).

Reads safetensors headers + raw bytes (no torch). Needs numpy, so it runs in the training image:
``uv run python -m vlm_pipeline.common.jobrun_cli --no-gpu ... -- python -m vlm_pipeline.train.weight_drift ...``.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DTYPES = {"F32": "<f4", "BF16": "<u2", "F16": "<f2", "I64": "<i8", "I32": "<i4", "U8": "u1", "BOOL": "u1"}


@dataclass(frozen=True)
class _Tensor:
    file: Path
    offset: int
    nbytes: int
    dtype: str


def _index(model_dir: Path) -> dict[str, _Tensor]:
    out: dict[str, _Tensor] = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        for key, meta in header.items():
            start, end = meta["data_offsets"]
            out[key] = _Tensor(path, 8 + n + start, end - start, meta["dtype"])
    return out


def _read(t: _Tensor) -> Any:
    import numpy as np

    with t.file.open("rb") as fh:
        fh.seek(t.offset)
        raw = fh.read(t.nbytes)
    return np.frombuffer(raw, dtype=_DTYPES[t.dtype])


def _as_f32_bits(t: _Tensor, values: Any) -> Any:
    """Float tensors as uint32 fp32 bit patterns (bf16 widened exactly); None for non-float tensors."""
    import numpy as np

    if t.dtype == "F32":
        return values.view(np.uint32)
    if t.dtype == "BF16":
        return values.astype(np.uint32) << 16
    if t.dtype == "F16":
        return values.astype(np.float32).view(np.uint32)
    return None


def _round_bf16(bits: Any) -> Any:
    """fp32 bit patterns rounded to the nearest bf16 (ties to even), as fp32 bit patterns."""
    import numpy as np

    wide = bits.astype(np.uint64)
    return (((wide + 0x7FFF + ((wide >> 16) & 1)) >> 16) << 16).astype(np.uint32)


def default_group(key: str) -> str:
    """Group = first two key parts, layer indices dropped: ``decoder.layers.3.fc1.weight`` -> ``decoder.layers``."""
    return ".".join(re.sub(r"\.\d+(\.|$)", r"\1", key).split(".")[:2])


@dataclass
class GroupDrift:
    group: str
    tensors: int = 0
    elements: int = 0
    moved: int = 0  # differ from base beyond bf16 rounding
    changed_bits: int = 0  # differ from base at all

    @property
    def moved_frac(self) -> float:
        return self.moved / self.elements if self.elements else 0.0

    @property
    def changed_bits_frac(self) -> float:
        return self.changed_bits / self.elements if self.elements else 0.0


def drift(base_dir: Path, ckpt_dir: Path) -> dict[str, GroupDrift]:
    """Per-group drift of ``ckpt_dir`` vs ``base_dir`` (same keys required; one tensor in memory at a time)."""
    base, ckpt = _index(base_dir), _index(ckpt_dir)
    if base.keys() != ckpt.keys():
        missing, extra = sorted(base.keys() - ckpt.keys()), sorted(ckpt.keys() - base.keys())
        raise ValueError(f"key sets differ: missing={missing[:5]} extra={extra[:5]}")
    groups: dict[str, GroupDrift] = {}
    for key in sorted(ckpt):
        g = groups.setdefault(default_group(key), GroupDrift(default_group(key)))
        b_t, c_t = base[key], ckpt[key]
        b_raw, c_raw = _read(b_t), _read(c_t)
        b_bits, c_bits = _as_f32_bits(b_t, b_raw), _as_f32_bits(c_t, c_raw)
        if b_raw.shape != c_raw.shape:
            raise ValueError(f"{key}: {b_t.dtype}{list(b_raw.shape)} != {c_t.dtype}{list(c_raw.shape)}")
        g.tensors += 1
        if b_bits is None or c_bits is None:  # integer buffers: exact compare
            g.elements += c_raw.size
            diff = int((b_raw != c_raw).sum())
            g.moved += diff
            g.changed_bits += diff
            continue
        g.elements += c_bits.size
        changed = b_bits != c_bits
        g.changed_bits += int(changed.sum())
        g.moved += int((changed & (_round_bf16(b_bits) != c_bits)).sum())
    return groups


def _iter_rows(groups: dict[str, GroupDrift]) -> Iterator[dict[str, Any]]:
    for g in groups.values():
        yield {**asdict(g), "moved_frac": round(g.moved_frac, 6), "changed_bits_frac": round(g.changed_bits_frac, 6)}


def check(
    groups: dict[str, GroupDrift], *, frozen: list[str], trainable: list[str], min_moved_frac: float
) -> list[str]:
    """Violations: a frozen group that moved, or a trainable group that moved less than ``min_moved_frac``.

    A prefix matches a group on dot boundaries (``encoder`` matches ``encoder.conv1``, not ``encoder_extra.x``).
    A prefix that matches no group is itself a violation, so a typo can never make the gate pass vacuously.
    """
    if not 0.0 <= min_moved_frac <= 1.0:
        raise ValueError(f"min_moved_frac must be in [0, 1], got {min_moved_frac}")

    def matches(group: str, prefix: str) -> bool:
        return group == prefix or group.startswith(prefix + ".")

    problems = [
        f"--{kind} prefix {p!r} matches no group"
        for kind, prefixes in (("frozen", frozen), ("trainable", trainable))
        for p in prefixes
        if not any(matches(g, p) for g in groups)
    ]
    for g in groups.values():
        if any(matches(g.group, p) for p in frozen) and g.moved:
            problems.append(f"frozen group {g.group} moved: {g.moved}/{g.elements}")
        if any(matches(g.group, p) for p in trainable) and g.moved_frac < min_moved_frac:
            problems.append(f"trainable group {g.group} barely moved: {g.moved_frac:.2%} < {min_moved_frac:.2%}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.train.weight_drift", description=__doc__.split("\n")[0])
    p.add_argument("--base", type=Path, required=True, help="base model dir (*.safetensors)")
    p.add_argument("--ckpt", type=Path, required=True, help="consolidated checkpoint dir (*.safetensors)")
    p.add_argument("--frozen", action="append", default=[], help="group prefix that must not move (repeatable)")
    p.add_argument("--trainable", action="append", default=[], help="group prefix that must move (repeatable)")
    p.add_argument("--min-moved-frac", type=float, default=0.1, help="minimum moved share of trainable groups")
    a = p.parse_args(argv)
    if not 0.0 <= a.min_moved_frac <= 1.0:
        p.error("--min-moved-frac must be in [0, 1]")
    groups = drift(a.base, a.ckpt)
    problems = check(groups, frozen=a.frozen, trainable=a.trainable, min_moved_frac=a.min_moved_frac)
    print(json.dumps({"groups": list(_iter_rows(groups)), "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
