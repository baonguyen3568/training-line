"""What a catalog download may fetch (security download-egress §3). Pure; shared by the download job and the host check.

The job filters the hub's file list with these rules before fetching anything; the host applies them AGAIN to what
landed in the staging dir before installing it (a job is not trusted to have followed them).
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable
from typing import Literal

Kind = Literal["model", "dataset"]

REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")

MODEL_ALLOW = (
    "*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "*processor_config.json",
    "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.*", "merges.txt", "*.model", "*.tiktoken",
    "chat_template.*", "*.jinja", "README.md", "LICENSE*", "NOTICE*",
)  # fmt: skip
DATASET_ALLOW = (
    "*.parquet", "*.arrow", "*.jsonl", "*.json", "*.csv", "*.png", "*.jpg", "*.jpeg", "*.webp", "README.md", "LICENSE*",
)  # fmt: skip
# WebDataset shards (ADR-021 A1, security §10): only for a user dataset whose format is webdataset, read by a
# network-none job member by member (download.archives); never for a catalog download.
ARCHIVE_ALLOW = ("*.tar", "*.tar.gz", "*.tgz", "*.zip")
IGNORE = (
    "*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle", "*.joblib", "*.npy", "*.npz", "*.h5", "*.msgpack",
    "*.gguf", "*.onnx", "*.zip", "*.7z", "*.tar.gz", "*.tgz", "*.sh", "*.exe", "*.so", ".git*", "*.py",
)  # fmt: skip


def _match(name: str, patterns: Iterable[str]) -> bool:
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(base, p) for p in patterns)


def name_ok(name: str) -> bool:
    """A relative POSIX file name inside the repo: no absolute path, no ``..``, no empty or dot parts, no control."""
    parts = name.split("/")
    return (
        0 < len(name) <= 512
        and not name.startswith("/")
        and "\\" not in name
        and all(p not in {"", ".", ".."} for p in parts)
        and not any(ord(c) < 32 or ord(c) == 127 for c in name)
    )


def wanted(name: str, kind: Kind, remote_code_files: Iterable[str] = (), *, archives: bool = False) -> bool:
    """Whether ``name`` may be fetched: allowed for its kind, never ignored; ``*.py`` only if listed for the repo."""
    if not name_ok(name):
        return False
    if kind == "model" and name in set(remote_code_files):  # configs/security allowlist, exact names only
        return True
    if archives and kind == "dataset" and _match(name, ARCHIVE_ALLOW):  # shards, checked again by the job
        return True
    if _match(name, IGNORE):  # ignore wins over allow
        return False
    return _match(name, MODEL_ALLOW if kind == "model" else DATASET_ALLOW)


def check_repo(repo: str, revision: str) -> None:
    """ValueError unless ``repo`` is ``org/name`` and ``revision`` a 40-hex commit (never a branch)."""
    if not REPO.fullmatch(repo) or not REVISION.fullmatch(revision):
        raise ValueError("repo must be org/name and revision a 40-hex commit")


__all__ = ["DATASET_ALLOW", "IGNORE", "MODEL_ALLOW", "Kind", "check_repo", "name_ok", "wanted"]
