"""Paths as the app sees them -> paths on the host, for the job mounts the runner asks docker for (ADR-015, M3).

In compose the app and the runner see the project at ``/app``, and VLM_DATA_ROOT is bind-mounted over
``/app/{.cache,outputs,data}``: app code keeps its project-relative paths (PM decision M3-b). A job container is
started by the host docker daemon, which needs HOST paths: ``.cache``/``outputs``/``data`` live under the data root,
everything else under the host project dir. Anything outside ``/app`` (or with ``..``) is refused, never guessed:
a mount source the app did not own must not reach docker.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DATA_DIRS = frozenset({".cache", "outputs", "data"})


class PathOutsideRootError(ValueError):
    """The path is not under the app root: no host path for it."""


@dataclass(frozen=True)
class HostPaths:
    app_root: PurePosixPath  # where the app/runner see the project (``/app`` in compose)
    host_project: PurePosixPath  # the same project on the host (code, configs)
    data_root: PurePosixPath | None = None  # VLM_DATA_ROOT on the host; None: data stays in the project dir

    def __post_init__(self) -> None:
        for name in ("app_root", "host_project", "data_root"):
            value = getattr(self, name)
            if value is not None and (not value.is_absolute() or ".." in value.parts):
                raise ValueError(f"{name} must be an absolute path without '..', got {value}")

    def to_host(self, path: str | PurePosixPath) -> PurePosixPath:
        p = PurePosixPath(path)
        if not p.is_absolute() or ".." in p.parts:
            raise PathOutsideRootError(f"{p}: only absolute paths without '..' can be mounted")
        try:
            rel = p.relative_to(self.app_root)
        except ValueError:
            raise PathOutsideRootError(f"{p} is outside {self.app_root}") from None
        if self.data_root is not None and rel.parts and rel.parts[0] in DATA_DIRS:
            return self.data_root / rel
        return self.host_project / rel


class UnsafeMountSourceError(ValueError):
    """A bind-mount source the daemon could be tricked into following (symlink), or not what the job expects."""


def check_mount_source(root: Path, source: Path, *, directory: bool) -> Path:
    """Refuse a bind-mount source unless every component from ``root`` down is a real (non-symlink) entry.

    The docker daemon follows a symlink in a SOURCE path (or in any of its parents) on the host: a job that planted
    ``outputs/runs/<A>/x -> ~/.ssh`` would get ``~/.ssh`` mounted into the next job reading A's output (bao-security,
    M3). A symlink INSIDE a mounted folder is harmless (resolved in the container). Checks, walking with
    ``O_PATH|O_NOFOLLOW`` from ``root`` (data root or host project, itself a real directory): no component is a
    symlink, every parent is a directory, the source exists (never let the daemon create it as root), has the
    expected kind and is owned by this user.

    Remaining TOCTOU between this check and the mount is accepted on one assumption: no running job can write a
    parent of another job's source (a job writes only its own, freshly created run dir).
    """
    if not root.is_absolute() or not source.is_absolute() or ".." in source.parts:
        raise UnsafeMountSourceError(f"{source}: absolute paths under {root} only")
    try:
        rel = source.relative_to(root)
    except ValueError:
        raise UnsafeMountSourceError(f"{source} is outside {root}") from None
    flags = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(root, flags | os.O_DIRECTORY)
    except OSError as exc:  # ELOOP / ENOTDIR: the root itself is a symlink or not a folder
        raise UnsafeMountSourceError(f"{root} is not a real directory ({exc.strerror})") from None
    try:
        for i, part in enumerate(rel.parts):
            try:
                st = os.stat(part, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                raise UnsafeMountSourceError(f"{source} does not exist") from None
            if stat.S_ISLNK(st.st_mode):
                raise UnsafeMountSourceError(f"{source}: {'/'.join(rel.parts[: i + 1])} is a symlink")
            last = i == len(rel.parts) - 1
            if not last:
                if not stat.S_ISDIR(st.st_mode):
                    raise UnsafeMountSourceError(f"{source}: {'/'.join(rel.parts[: i + 1])} is not a directory")
                nxt = os.open(part, flags | os.O_DIRECTORY, dir_fd=fd)
                os.close(fd)
                fd = nxt
        st = os.fstat(fd) if not rel.parts else st
    finally:
        os.close(fd)
    kind_ok = stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)
    if not kind_ok:
        raise UnsafeMountSourceError(f"{source} is not a {'directory' if directory else 'regular file'}")
    if st.st_uid != os.getuid():
        raise UnsafeMountSourceError(f"{source} is not owned by this user")
    return source


__all__ = ["DATA_DIRS", "HostPaths", "PathOutsideRootError", "UnsafeMountSourceError", "check_mount_source"]
