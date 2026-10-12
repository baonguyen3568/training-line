"""Project paths and size units shared by every module (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path

GIB = 1024**3  # every ``*_gb`` value in this project is GiB (like ``free -g``; ``df -H`` prints powers of 1000)
KIB_PER_GIB = 1024 * 1024

PROJECT_ROOT_ENV = "VLM_PROJECT_ROOT"


def project_root(start: Path | None = None) -> Path:
    """Repository root: ``$VLM_PROJECT_ROOT`` if set, else the nearest parent holding ``pyproject.toml``.

    Works for editable installs and for ``PYTHONPATH=/src`` inside containers (set the env var there).
    """
    env = os.environ.get(PROJECT_ROOT_ENV)
    if env:
        return Path(env).resolve()
    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"project root not found from {here}; set {PROJECT_ROOT_ENV}")
