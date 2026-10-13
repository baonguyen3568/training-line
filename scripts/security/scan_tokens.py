"""Token leak gate (PM 2026-10-13: the user keeps a write-scoped HF token, so leaks must be caught mechanically).

    uv run python scripts/security/scan_tokens.py tracked            # every file tracked by git (CI gate)
    uv run python scripts/security/scan_tokens.py paths outputs/ ... # runtime artefacts: logs, run dirs, exports

Matches the *shape* of real credentials, not the bare ``hf_`` prefix (``hf_hub``, ``hf_model.py`` are code):
Hugging Face user/org tokens (``hf_`` + 30-40 alphanumerics with both cases; current tokens are 34 letters, no digit),
NVIDIA API keys (``nvapi-``) and W&B keys in assignments. Findings print the file, line and a redacted value
(first 6 characters); the secret itself is never printed. Exit 1 on any finding, 2 on usage errors. Stdlib only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vlm_pipeline.common.secret_scan import PATTERNS, Leak, scan_files, scan_text, walk  # noqa: F401 - re-exported


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607
        capture_output=True,
        check=True,
    ).stdout
    return [root / p for p in out.decode().split("\0") if p]


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
