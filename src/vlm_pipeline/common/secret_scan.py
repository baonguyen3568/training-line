"""Credential-shaped strings (bao-security token gate), shared by the CI script and by runtime checks (exports).

Matches the *shape* of real credentials, not the bare ``hf_`` prefix: Hugging Face tokens (``hf_`` + 30-40
alphanumerics with both cases), NVIDIA API keys (``nvapi-``) and W&B keys in assignments. A finding carries a redacted
value (first 6 characters); the secret itself is never returned or printed. Stdlib only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PATTERNS = {
    "HF_TOKEN": re.compile(
        r"(?<![A-Za-z0-9_])hf_(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])[A-Za-z0-9]{30,40}(?![A-Za-z0-9])"
    ),
    "NGC_KEY": re.compile(r"(?<![A-Za-z0-9])nvapi-[A-Za-z0-9_-]{40,}"),
    "WANDB_KEY": re.compile(r"WANDB_API_KEY\s*[=:]\s*['\"]?[0-9a-f]{40}\b"),
}
MAX_BYTES = 50 * 1024 * 1024  # skip larger files (weights); tokens live in text, logs and configs
SKIP_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".engine", ".arrow", ".parquet"}


@dataclass(frozen=True)
class Leak:
    path: str
    line: int
    kind: str
    redacted: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} {self.redacted}…"


def scan_text(path: str, text: str) -> Iterator[Leak]:
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            for m in pattern.finditer(line):
                yield Leak(path, lineno, kind, m.group(0)[:6])


def scan_files(files: Iterable[Path], *, base: Path) -> list[Leak]:
    leaks: list[Leak] = []
    for f in files:
        if f.is_symlink() or not f.is_file() or f.suffix.lower() in SKIP_SUFFIXES or f.stat().st_size > MAX_BYTES:
            continue
        text = f.read_bytes().decode("utf-8", errors="replace")
        rel = f.relative_to(base).as_posix() if f.is_relative_to(base) else f.as_posix()
        leaks += scan_text(rel, text)
    return leaks


def walk(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            yield from (f for f in p.rglob("*") if not f.is_symlink())
        else:
            yield p


__all__ = ["MAX_BYTES", "PATTERNS", "SKIP_SUFFIXES", "Leak", "scan_files", "scan_text", "walk"]
