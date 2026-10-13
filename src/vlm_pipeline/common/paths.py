"""Project paths and size units shared by every module (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path

GIB = 1024**3  # every ``*_gb`` value in this project is GiB (like ``free -g``; ``df -H`` prints powers of 1000)
KIB_PER_GIB = 1024 * 1024

PROJECT_ROOT_ENV = "VLM_PROJECT_ROOT"
CODE_ROOT_ENV = "VLM_CODE_ROOT"  # a clean worktree of HEAD whose src/ + configs/ jobs run (research/E2E runs)


def project_root(start: Path | None = None) -> Path:
    """Repository root: ``$VLM_PROJECT_ROOT`` if set, else the nearest parent holding ``pyproject.toml``.

    Works for editable installs and for ``PYTHONPATH=/src`` inside containers (set the env var there).
    """
    env = os.environ.get(PROJECT_ROOT_ENV)
    if env:
        root = Path(env).resolve()
        # A wrong env var must never point delete/write code at an unrelated directory.
        if not _is_project_root(root):
            raise FileNotFoundError(f"${PROJECT_ROOT_ENV}={env} is not the vlm-pipeline repository")
        return root
    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        if _is_project_root(parent):
            return parent
    raise FileNotFoundError(f"project root not found from {here}; set {PROJECT_ROOT_ENV}")


def _is_project_root(path: Path) -> bool:
    try:
        text = (path / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return False
    return 'name = "vlm-pipeline"' in text


def is_team_worktree_dir(root: Path, code: Path) -> bool:
    """``code`` sits where the team keeps worktrees of ``root`` and looks like one (no git call; jobrun adds that).

    Under ``.cache/wt-*``, ``.cache/check-clean.*`` or ``.worktrees/*`` only (never outputs/, data/, uploads/: a
    job could plant pyproject.toml + configs/ there), not a symlink, with a ``.git`` FILE (worktree link).
    """
    if not code.is_relative_to(root):
        return False
    parts = code.relative_to(root).parts
    placed = len(parts) == 2 and (
        (parts[0] == ".cache" and parts[1].startswith(("wt-", "check-clean."))) or parts[0] == ".worktrees"
    )
    return placed and not code.is_symlink() and (code / ".git").is_file() and _is_project_root(code)


def configs_root(root: Path | None = None) -> Path:
    """Where the host reads ``configs/`` (catalogs, templates, standards, licences): the code the jobs run.

    ``$VLM_CODE_ROOT`` (a team worktree of ``root``) when set, so the app half and the job half of a run never see
    two different catalogs (research 2026-10-13); else ``root`` (default: the project root). Set but not a team
    worktree: ValueError (security: never a silent fallback to the root's configs). Data, outputs and the GPU lock
    always stay ``root``'s.
    """
    base = (root if root is not None else project_root()).resolve()
    raw = os.environ.get(CODE_ROOT_ENV, "").strip()
    if not raw:
        return base
    code = Path(raw).resolve()
    if code == base:
        return base
    if not (is_team_worktree_dir(base, code) and (code / "configs").is_dir()):  # fail closed, like with_code_root
        raise ValueError(
            f"${CODE_ROOT_ENV} must be a git worktree checkout of this project (.cache/wt-*, .worktrees/*)"
        )
    return code
