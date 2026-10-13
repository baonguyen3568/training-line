"""Open ``<run_dir>/logs/<name>`` without ever following a link (bao-security R2), for every host-side log reader.

The run dir belongs to the job, and model remote code may run inside it: ``logs`` may have been swapped for a symlink
(to ``~/.config/ntc-training``, ``.env``, the HF token...) and the log for a FIFO. Each step goes through a dir fd
with ``O_NOFOLLOW``; the file is opened ``O_NONBLOCK`` (a FIFO never blocks the reader) and must be a regular file.
The one opener of run logs: vlm-runner and the app workers both use it.
"""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path


class RunLogError(ValueError):
    """The run log cannot be used safely (a link, a FIFO, missing...): its content must not be read."""


def open_run_log(run_dir: Path, *, name: str = "stdout.log", write: bool = False) -> int:
    """The fd of ``<run_dir>/logs/<name>``, reached through dir fds, never through a link; RunLogError otherwise."""
    if "/" in name or name in {"", ".", ".."}:
        raise RunLogError(f"bad log name {name!r}")
    try:
        run_fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if write:
                with contextlib.suppress(FileExistsError):
                    os.mkdir("logs", 0o755, dir_fd=run_fd)
            logs_fd = os.open("logs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=run_fd)
        finally:
            os.close(run_fd)
        try:
            flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT) if write else os.O_RDONLY
            fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o644, dir_fd=logs_fd)
        finally:
            os.close(logs_fd)
    except OSError as exc:  # ELOOP / ENOTDIR (a link), ENXIO (a FIFO with no reader), ENOENT...
        raise RunLogError(f"run log unusable ({exc.strerror})") from None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RunLogError(f"{name} is not a regular file")
    return fd


def tail_run_log(run_dir: Path, lines: int, *, name: str = "stdout.log", max_bytes: int = 256 * 1024) -> list[str]:
    """The last ``lines`` lines (from at most the last ``max_bytes``); RunLogError when the log is unsafe/missing."""
    fd = open_run_log(run_dir, name=name)
    with os.fdopen(fd, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        f.seek(max(0, size - max_bytes))
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


def log_tail(run_dir: Path, n: int = 30) -> str:
    """The last ``n`` lines of ``<run_dir>/logs/stdout.log`` for an error card; "" when the log is unsafe or missing."""
    try:
        return "\n".join(tail_run_log(run_dir, n))
    except RunLogError:
        return ""


__all__ = ["RunLogError", "log_tail", "open_run_log", "tail_run_log"]
