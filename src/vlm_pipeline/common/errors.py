"""Shared error types and process exit codes.

Stdlib only (container workers import ``ExitCode`` via PYTHONPATH); the catalog is loaded by
``common.error_catalog``. Exceptions carry only a catalog ``code`` plus ``params``; user-facing text lives in
``i18n/{vi,en}.yaml`` and is rendered at the app edge (Proposal 007 R9). Codes follow ``<GROUP>-<NNN>``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import ClassVar


class ExitCode(IntEnum):
    """Process exit codes shared by preflight, job runner, worker and dashboard (ADR-005 D8, Proposal 007)."""

    OK = 0
    NO_MEM = 2
    NO_DISK = 3
    LOCK_BUSY = 4
    MEM_PREEMPT = 75  # worker stopped by the memory watchdog after checkpointing -> run status "interrupted"


class VlmPipelineError(Exception):
    """Base class for every error raised by vlm_pipeline."""

    code: ClassVar[str] = "SYS-000"

    def __init__(self, message: str = "", *, params: Mapping[str, object] | None = None) -> None:
        super().__init__(message or self.code)
        self.params: dict[str, object] = dict(params or {})


class InsufficientMemoryError(VlmPipelineError):
    """Not enough unified memory to start a job (preflight refused)."""

    code: ClassVar[str] = "MEM-001"
