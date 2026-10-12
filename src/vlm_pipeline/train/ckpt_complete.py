"""Make an exported HF checkpoint contain every tensor of the base checkpoint it was trained from (host-safe, stdlib).

BUG-002 (bao-tester, O-1b parity): AutoModel's consolidated export keeps only keys of the live model's state_dict.
Nemotron-Parse loses three keys of NVIDIA's checkpoint that way:
- ``...radio_model.input_conditioner.norm_mean/norm_std``: RADIO's ``make_preprocessor_external()`` swaps the
  conditioner for ``nn.Identity`` at the first forward, so the buffers vanish from the state_dict. NVIDIA's modeling
  code then loads them as 0 and any path that runs the conditioner divides by zero;
- ``decoder.embed_positions.weight``: present in NVIDIA's file but used by no modeling code. Copied anyway (harmless,
  and "export contains the base" then holds literally, so a strict loader or another runtime finds what it expects).
Keys missing from the export are copied byte-for-byte from the base into one extra shard and added to the index, then
the export is checked to be a superset of the base. Tensors the export already has are never touched.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
EXTRA_SHARD = "model-vlm-restored-from-base.safetensors"
# Keys that are an alias of another tensor when embeddings are tied: restoring the base copy would bring back a stale
# head next to the trained embedding.
TIED_ALIASES = frozenset({"lm_head.weight"})
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class CheckpointIncompleteError(RuntimeError):
    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = sorted(missing)
        super().__init__(f"exported checkpoint lacks {len(self.missing)} base tensor(s): {self.missing[:5]}")


def read_header(path: Path) -> tuple[dict[str, dict[str, object]], int]:
    """Tensor entries of a .safetensors file and the byte offset where its data starts."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        (size,) = struct.unpack("<Q", raw)
        if size > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: safetensors header of {size} bytes")
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def weight_map(model_dir: Path) -> dict[str, str]:
    """Tensor name -> shard file name for an HF checkpoint dir (sharded with an index, or a single file)."""
    index = model_dir / INDEX_NAME
    if index.is_file():
        return dict(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    single = model_dir / SINGLE_NAME
    if single.is_file():
        return dict.fromkeys(read_header(single)[0], SINGLE_NAME)
    raise FileNotFoundError(f"no {INDEX_NAME} or {SINGLE_NAME} in {model_dir}")


def _tied(model_dir: Path) -> bool:
    path = model_dir / "config.json"
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding="utf-8"))
    subs = [config] + [config[k] for k in ("text_config", "decoder") if isinstance(config.get(k), Mapping)]
    return any(bool(c.get("tie_word_embeddings", False)) for c in subs)


def missing_from_export(base_dir: Path, export_dir: Path) -> list[str]:
    """Base tensors the export lacks, ignoring tied aliases when the export ties its embeddings."""
    missing = set(weight_map(base_dir)) - set(weight_map(export_dir))
    if _tied(export_dir):
        missing -= TIED_ALIASES
    return sorted(missing)


def _write_shard(path: Path, tensors: Mapping[str, tuple[dict[str, object], bytes]]) -> int:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (entry, data) in tensors.items():
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # data must start 8-byte aligned (safetensors spec)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)) + blob)
        for _, data in tensors.values():
            fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    return offset


def _span(entry: Mapping[str, object]) -> tuple[int, int]:
    offsets = entry["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"bad data_offsets {offsets!r}")
    return int(offsets[0]), int(offsets[1])


def _read_tensor(shard: Path, name: str) -> tuple[dict[str, object], bytes]:
    header, start = read_header(shard)
    entry = header[name]
    begin, end = _span(entry)
    if not 0 <= begin <= end or start + end > shard.stat().st_size:
        raise ValueError(f"{shard}: bad data_offsets for {name}")
    with shard.open("rb") as fh:
        fh.seek(start + begin)
        return entry, fh.read(end - begin)


def complete_from_base(base_dir: Path, export_dir: Path) -> list[str]:
    """Copy base tensors missing from ``export_dir`` into it; return the restored names. Idempotent.

    Raises ``CheckpointIncompleteError`` if the export is still not a superset of the base afterwards.
    """
    missing = missing_from_export(base_dir, export_dir)
    if missing:
        base_map = weight_map(base_dir)
        tensors = {name: _read_tensor(base_dir / base_map[name], name) for name in missing}
        extra = export_dir / EXTRA_SHARD
        kept = 0
        if extra.is_file():  # keep what an earlier call restored: the shard is rewritten as a whole
            for name in read_header(extra)[0]:
                if name not in tensors:
                    tensors[name] = _read_tensor(extra, name)
                    kept += len(tensors[name][1])
        index_path = export_dir / INDEX_NAME
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:  # single-file export: an index is needed to reference the extra shard
            header, _ = read_header(export_dir / SINGLE_NAME)
            data = sum(_span(e)[1] - _span(e)[0] for e in header.values())
            index = {"metadata": {"total_size": data}, "weight_map": dict.fromkeys(header, SINGLE_NAME)}
        added = _write_shard(extra, tensors) - kept  # bytes not yet counted in total_size
        metadata = index.setdefault("metadata", {})
        metadata["total_size"] = int(metadata.get("total_size", 0)) + added
        metadata["vlm_restored_from_base"] = sorted(set(metadata.get("vlm_restored_from_base", [])) | set(missing))
        index["weight_map"].update(dict.fromkeys(missing, EXTRA_SHARD))
        tmp = index_path.with_name(index_path.name + ".tmp")
        tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        tmp.replace(index_path)
    still = missing_from_export(base_dir, export_dir)
    if still:
        raise CheckpointIncompleteError(still)
    return missing


__all__ = [
    "CheckpointIncompleteError",
    "complete_from_base",
    "missing_from_export",
    "read_header",
    "weight_map",
]
