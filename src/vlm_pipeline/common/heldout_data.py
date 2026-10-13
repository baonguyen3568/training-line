"""Benchmark test data extracted to files (``data/eval/``) is never training data (tester/security, release 1.0).

Eval-only benchmarks (OCRBench...) are written under ``data/eval/<name>-<sha8>/`` so the vLLM gate can read them as a
folder. To ``data.local_folder`` that folder looks like any other dataset; this module is the one place that says
"this folder is held out": graph validate (rule DATA-007) and the train worker (run time) both ask it.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import ClassVar

from vlm_pipeline.common.errors import VlmPipelineError

HELDOUT_DATA_REL = PurePosixPath("data") / "eval"


def is_heldout_folder(root: Path | None, path: str | Path) -> bool:
    """True when ``path`` (relative to ``root``, or absolute) is ``data/eval`` or inside it, after symlinks.

    Without a root (validate with no workspace) the check is on the normalized text only.
    """
    if root is None:
        norm = PurePosixPath(os.path.normpath(str(path)))
        return norm == HELDOUT_DATA_REL or HELDOUT_DATA_REL in norm.parents
    base = Path(os.path.realpath(root))
    heldout = Path(os.path.realpath(base / HELDOUT_DATA_REL))
    target = Path(os.path.realpath(base / path))  # an absolute path stays absolute
    return target == heldout or heldout in target.parents


class HeldoutDataInTrainingError(VlmPipelineError):
    """Run-time twin of graph rule DATA-007: the train/val data of a fine-tune job comes from ``data/eval/``."""

    code: ClassVar[str] = "DATA-007"
