"""Runtime errors of graph nodes and their classification into error-catalog codes (P009 §5, Proposal 007 R9).

Domain objects carry ``code`` + ``params`` only; ``details`` is technical, redacted, and shown on demand.
Classification order (agreed with bao-backend/bao-tester, P009 §5):
cancelled by the executor -> interrupted, no error | exit 75 / memguard -> MEM-002 interrupted |
exit 137 + OOMKilled -> MEM-003 | error event with a catalog code -> that code | other non-zero exit -> SYS-003 |
catalog exception -> its code | anything else -> SYS-000.
"""

from __future__ import annotations

import re
import signal
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from vlm_pipeline.common.error_catalog import FALLBACK_CODE, load_catalog, log_matches
from vlm_pipeline.common.errors import ExitCode, VlmPipelineError

NodeOutcome = Literal["interrupted", "failed"]


# -------------------------------------------------------------------------------------------- exceptions


class InterruptedByMemguard(VlmPipelineError):  # noqa: N818 - name fixed by the spec (P009 §5)
    """Worker stopped by the memory watchdog after checkpointing (exit 75)."""

    code: ClassVar[str] = "MEM-002"


class OutOfMemoryUMA(VlmPipelineError):  # noqa: N818 - name fixed by the spec (P009 §5)
    """CUDA / unified-memory OOM inside a node."""

    code: ClassVar[str] = "MEM-003"


class ContainerExit(Exception):  # noqa: N818 - mirrors the container state, not an error by itself
    """A container job ended with a non-zero exit (raised by the job runner, or by FakeWorker in tests)."""

    def __init__(
        self,
        exit_code: int,
        oom_killed: bool = False,
        events: Iterable[Mapping[str, object]] = (),
        cancelled_by_executor: bool = False,
        log_tail: str = "",
    ) -> None:
        super().__init__(f"container exited with {exit_code}")
        self.exit_code = exit_code
        self.oom_killed = oom_killed
        self.events = tuple(events)
        self.cancelled_by_executor = cancelled_by_executor
        self.log_tail = log_tail


# ------------------------------------------------------------------------------------------- user error


@dataclass(frozen=True)
class UserError:
    """What the UI shows for a failed/interrupted node (text from i18n ``error.<code>.*``)."""

    code: str
    params: Mapping[str, object] = field(default_factory=dict)
    details: str = ""  # redacted technical detail ("Chi tiết kỹ thuật")
    resumable: bool = False  # a real checkpoint exists on disk
    checkpoint_step: int | None = None


# ------------------------------------------------------------------------------------------------ redact

_SECRET = re.compile(
    r"hf_[A-Za-z0-9]{8,}|nvapi-[A-Za-z0-9_-]{8,}|(?<![A-Za-z])sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}"
)
_HOME = re.compile(r"/home/[^/\s'\"]+|/root(?=/)|/Users/[^/\s'\"]+")
_KV_SECRET = re.compile(r"(?i)\b(token|password|passwd|secret|api[_-]?key)\s*[=:]\s*\S+")


def redact(text: str) -> str:
    """Remove secrets and home paths from technical text before it is stored or shown."""
    text = _SECRET.sub("<redacted>", text)
    text = _KV_SECRET.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    return _HOME.sub("~", text)


def _details(exc: BaseException, max_lines: int = 30) -> str:
    lines = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).splitlines()
    return redact("\n".join(lines[-max_lines:]))


# --------------------------------------------------------------------------------------------- classify


@dataclass(frozen=True)
class Classified:
    status: NodeOutcome
    error: UserError | None  # None: not an error (user cancel)


def _signal_name(exit_code: int) -> str | None:
    if exit_code > 128:
        try:
            return signal.Signals(exit_code - 128).name
        except ValueError:
            return None
    return None


def _event_error(events: Iterable[Mapping[str, object]]) -> tuple[str, Mapping[str, object]] | None:
    catalog = load_catalog()
    for event in reversed(tuple(events)):
        code = event.get("code")
        if event.get("type") == "error" and isinstance(code, str) and code in catalog:
            params = event.get("params")
            return code, params if isinstance(params, Mapping) else {}
    return None


def _classify_container_exit(exc: ContainerExit, cancelled: bool, checkpoint_step: int | None) -> Classified:
    if exc.cancelled_by_executor or (cancelled and exc.exit_code in (137, 143)):
        return Classified("interrupted", None)
    details = redact(exc.log_tail) if exc.log_tail else _details(exc)

    def err(code: str, params: Mapping[str, object] | None = None) -> UserError:
        return UserError(code, dict(params or {}), details, checkpoint_step is not None, checkpoint_step)

    if exc.exit_code == ExitCode.MEM_PREEMPT:
        return Classified("interrupted", err("MEM-002"))
    if exc.exit_code == 137 and exc.oom_killed:  # 128 + SIGKILL
        return Classified("failed", err("MEM-003"))
    if (found := _event_error(exc.events)) is not None:
        return Classified("failed", err(*found))
    if exc.log_tail and log_matches("MEM-003", exc.log_tail):  # e.g. "CUDA error: out of memory" at model load
        return Classified("failed", err("MEM-003"))
    return Classified("failed", err("SYS-003", {"exit_code": exc.exit_code, "signal": _signal_name(exc.exit_code)}))


def classify(exc: BaseException, *, cancelled: bool, checkpoint_step: int | None) -> Classified:
    """Map a worker exception to the node outcome and the error shown to the user."""
    if isinstance(exc, ContainerExit):
        return _classify_container_exit(exc, cancelled, checkpoint_step)
    if cancelled:
        return Classified("interrupted", None)
    resumable = checkpoint_step is not None
    if isinstance(exc, InterruptedByMemguard):
        return Classified("interrupted", UserError("MEM-002", {}, _details(exc), resumable, checkpoint_step))
    code = exc.code if isinstance(exc, VlmPipelineError) and exc.code in load_catalog() else FALLBACK_CODE
    params = exc.params if code != FALLBACK_CODE and isinstance(exc, VlmPipelineError) else {}
    return Classified("failed", UserError(code, dict(params), _details(exc), resumable, checkpoint_step))


__all__ = [
    "Classified",
    "ContainerExit",
    "InterruptedByMemguard",
    "OutOfMemoryUMA",
    "UserError",
    "classify",
    "redact",
]
