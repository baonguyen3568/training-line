"""How a Hugging Face dataset repo lays out its data files (ADR-021 / P021 H4): configs, splits, format, a sample.

Pure functions over the repo's file list, shared by the resolve job (which sample to fetch) and the host. Only the
file layouts the hub itself documents are read, never a loading script:

* ``data/<split>-00000-of-00001.parquet`` (the default config), ``<config>/<split>-*.parquet``;
* the hub's parquet export ``<config>/<split>/0000.parquet`` (``refs/convert/parquet``);
* ``<split>.jsonl`` / ``<split>.json`` / ``<split>.csv`` / ``*.arrow`` at any of those places;
* imagefolder: ``<split>/metadata.jsonl`` (or ``.csv``) next to the images of that split.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

DATA_SUFFIXES = (".parquet", ".arrow", ".jsonl", ".json", ".csv")
SHARD_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")  # WebDataset (download.archives reads them)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
FORMAT_ORDER = ("parquet", "arrow", "jsonl", "json", "csv", "webdataset")  # the sample prefers the first present
SAMPLE_MAX = 256 << 20  # one data file fetched to inspect a split (a parquet footer + its first row group)
SAMPLE_IMAGES = 3  # imagefolder: metadata + this many images
_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "validation": "validation", "valid": "validation", "val": "validation", "dev": "validation",
    "test": "test", "testing": "test", "eval": "test",
}  # fmt: skip
_SPLIT_WORD = re.compile(r"(?:^|[-_.])(train|training|validation|valid|val|dev|test|testing|eval)(?=[-_.0-9]|$)")
_NOT_DATA = ("README.md", ".gitattributes")


@dataclass(frozen=True)
class DataFile:
    name: str
    config: str
    split: str
    format: str  # parquet | arrow | jsonl | json | csv | imagefolder


def _split_of(parts: list[str]) -> str | None:
    """The split a path names: a directory or the file's own name, by the hub's conventions."""
    for part in reversed(parts):  # "train-0000.tar", "docvqa-train-0001.tar", "data/validation/0.parquet"
        match = _SPLIT_WORD.search(part.lower())
        if match:
            return _SPLIT_ALIASES[match.group(1)]
    return None


def classify(name: str) -> DataFile | None:
    """The data file ``name`` is, or None (a README, a script, an image, something unknown)."""
    lower = name.lower()
    if name in _NOT_DATA or not lower.endswith((*DATA_SUFFIXES, *SHARD_SUFFIXES)):
        return None
    parts = name.split("/")
    split = _split_of(parts)
    if split is None:
        split = "train"  # a lone data file: the hub loads it as the train split
    config = parts[0] if len(parts) > 1 and parts[0] != "data" and _split_of([parts[0]]) is None else "default"
    stem = parts[-1].lower()
    if stem.startswith("metadata.") and len(parts) >= 1:
        return DataFile(name, config, split, "imagefolder")
    if lower.endswith(SHARD_SUFFIXES):
        return DataFile(name, config, split, "webdataset")
    fmt = next(s.lstrip(".") for s in DATA_SUFFIXES if lower.endswith(s))
    return DataFile(name, config, split, fmt)


def layout(names: Iterable[str]) -> dict[str, dict[str, list[DataFile]]]:
    """``{config: {split: [files]}}`` of every data file the hub layouts describe."""
    out: dict[str, dict[str, list[DataFile]]] = {}
    for name in sorted(names):
        f = classify(name)
        if f is not None:
            out.setdefault(f.config, {}).setdefault(f.split, []).append(f)
    return out


def has_script(names: Iterable[str]) -> bool:
    """A loading script anywhere (the app never runs one: DATA-009)."""
    return any(n.lower().endswith(".py") for n in names)


def choose(tree: Mapping[str, Mapping[str, list[DataFile]]], config: str | None, split: str | None) -> tuple[str, str]:
    """The (config, split) to inspect: the asked ones when they exist, else default/first config and train/first."""
    cfg = config if config is not None and config in tree else ("default" if "default" in tree else sorted(tree)[0])
    splits = tree[cfg]
    if split is not None and split in splits:
        return cfg, split
    return cfg, "train" if "train" in splits else sorted(splits)[0]


def sample_files(files: list[DataFile], sizes: Mapping[str, int], all_names: Iterable[str]) -> tuple[str, list[str]]:
    """(format, files to fetch) for one split: the smallest file of the preferred format, under SAMPLE_MAX."""
    meta = [f for f in files if f.format == "imagefolder"]
    if meta:
        folder = meta[0].name.rsplit("/", 1)[0] + "/" if "/" in meta[0].name else ""
        images = sorted(n for n in all_names if n.startswith(folder) and n.lower().endswith(IMAGE_SUFFIXES))
        picked = [meta[0].name, *images[:SAMPLE_IMAGES]]
        return "imagefolder", [n for n in picked if 0 <= sizes.get(n, SAMPLE_MAX + 1) <= SAMPLE_MAX]
    for fmt in FORMAT_ORDER:
        candidates = sorted((sizes.get(f.name, SAMPLE_MAX + 1), f.name) for f in files if f.format == fmt)
        if candidates and candidates[0][0] <= SAMPLE_MAX:
            return fmt, [candidates[0][1]]
        if candidates:
            return fmt, []  # there, but too big to sample: the inspect says so, never guesses
    return "unknown", []


__all__ = ["DATA_SUFFIXES", "SAMPLE_MAX", "DataFile", "choose", "classify", "has_script", "layout", "sample_files"]
