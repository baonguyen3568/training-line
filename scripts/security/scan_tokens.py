"""Token leak gate (PM 2026-10-13: the user keeps a write-scoped HF token, so leaks must be caught mechanically).

    uv run python scripts/security/scan_tokens.py tracked            # every file tracked by git (CI gate)
    uv run python scripts/security/scan_tokens.py paths outputs/ ... # runtime artefacts: logs, run dirs, exports

Matches the *shape* of real credentials, not the bare ``hf_`` prefix (``hf_hub``, ``hf_model.py`` are code):
Hugging Face user/org tokens (``hf_`` + 30-40 alphanumerics with both cases; current tokens are 34 letters, no digit),
NVIDIA API keys (``nvapi-``) and W&B keys in assignments. Findings print the file, line and a redacted value
(first 6 characters); the secret itself is never printed. Exit 1 on any finding, 2 on usage errors. Stdlib only.
"""

from __future__ import annotations

import re
import subprocess
import sys
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


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout
    return [root / p for p in out.decode().split("\0") if p]


def walk(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            yield from (f for f in p.rglob("*") if not f.is_symlink())
        else:
            yield p


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path.cwd()
    if args[:1] == ["tracked"] and len(args) == 1:
        leaks = scan_files(tracked_files(root), base=root)
    elif args[:1] == ["paths"] and len(args) > 1:
        leaks = scan_files(walk(Path(a) for a in args[1:]), base=root)
    else:
        print(__doc__, file=sys.stderr)
        return 2
    for leak in leaks:
        print(leak)
    print(f"scan_tokens: {len(leaks)} finding(s)")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
