"""Paths as the app sees them -> paths on the host, for the job mounts the runner asks docker for (ADR-015, M3).

In compose the app and the runner see the project at ``/app``, and VLM_DATA_ROOT is bind-mounted over
``/app/{.cache,outputs,data}``: app code keeps its project-relative paths (PM decision M3-b). A job container is
started by the host docker daemon, which needs HOST paths: ``.cache``/``outputs``/``data`` live under the data root,
everything else under the host project dir. Anything outside ``/app`` (or with ``..``) is refused, never guessed:
a mount source the app did not own must not reach docker.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DATA_DIRS = frozenset({".cache", "outputs", "data", "uploads"})


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


def _internal_link(link: Path, top: Path) -> bool:
    """A relative symlink whose target exists inside ``top`` (resolved on the host, the way the container will)."""
    return not link.readlink().is_absolute() and link.exists() and link.resolve().is_relative_to(top)


def tree_problem(root: Path, *, allow_internal_symlinks: bool = False) -> str | None:
    """The first entry under ``root`` a container must not get, or None (P017 C3, bao-security step 6 Q1).

    A container follows symlinks INSIDE a read-only mount, so before ``root`` is mounted every entry must be a plain
    dir or regular file. ``allow_internal_symlinks`` (HF hub repo dirs: ``snapshots/<rev>/f -> ../../blobs/<sha>``)
    also accepts a RELATIVE symlink that resolves inside ``root``; an absolute one, or one leading out, is refused.
    A dir that cannot be listed or an entry that vanishes while walking cannot be verified: refused.
    """
    top = root.resolve()
    unreadable: list[str] = []

    def failed(err: OSError) -> None:
        unreadable.append(str(err.filename))

    try:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=failed):
            for name in (*dirnames, *filenames):
                entry = Path(dirpath) / name
                mode = entry.lstat().st_mode
                if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
                    continue
                if allow_internal_symlinks and stat.S_ISLNK(mode) and _internal_link(entry, top):
                    continue
                return entry.relative_to(root).as_posix()
    except OSError as exc:
        return f"unreadable: {exc.strerror}"
    return f"unreadable: {unreadable[0]}" if unreadable else None


MAX_LINK_HOPS = 8


def _hub_unit(root: Path, hop: Path) -> Path | None:
    """The mount unit a link hop needs (P017 step 6): its HF repo dir, or one file of the shared blob store."""
    if not hop.is_relative_to(root):
        return None
    rel = hop.relative_to(root)
    if rel.parts[:3] != (".cache", "hf", "hub") or len(rel.parts) < 4:
        return None
    if rel.parts[3] == "blobs":  # shared xet/blob store: only the very file, never the whole store
        return hop if len(rel.parts) > 4 else None
    if not re.match(r"^(?:models|datasets|spaces)--", rel.parts[3]):  # a stray file (version.txt...) is no unit
        return None
    return root.joinpath(*rel.parts[:4])


SECRET_NAMES = re.compile(
    r"^(?:\.env.*|token|stored_tokens|.*\.pem|.*\.key|id_[a-z0-9]+|\.netrc|\.git-credentials|credentials.*\.json)$"
)


def secret_file_problem(top: Path) -> str | None:
    """A file under ``top`` that looks like a credential (HF login token, .env, keys): never mounted (S6-1)."""
    for dirpath, dirs, files in os.walk(top, followlinks=False):
        for name in (*files, *dirs):
            if SECRET_NAMES.fullmatch(name):
                return (Path(dirpath) / name).relative_to(top).as_posix()
    return None


def link_units(root: Path, top: Path) -> tuple[str | None, set[Path]]:
    """(problem, extra mount units) for the symlinks under ``top`` (HF caches; security step 6 Q1).

    Every link is followed hop by hop on the host: each hop must be RELATIVE and land inside ``top`` itself or in
    ``<root>/.cache/hf/hub`` (one repo dir, or one file of the shared blob store), the chain ends on a regular file
    and is at most ``MAX_LINK_HOPS`` long. The units it passes through are returned, so the container can resolve the
    same chain inside its read-only mounts. Anything else (absolute link, a hop leading elsewhere) is a problem.
    """
    units: set[Path] = set()
    top = Path(os.path.normpath(top))
    for dirpath, _dirs, files in os.walk(top, followlinks=False):
        for name in files + [d for d in _dirs if (Path(dirpath) / d).is_symlink()]:
            link = Path(dirpath) / name
            if not link.is_symlink():
                continue
            hop = link
            for _ in range(MAX_LINK_HOPS):
                target = hop.readlink()
                if target.is_absolute():
                    return f"{link.relative_to(top)}: absolute link", units
                hop = Path(os.path.normpath(hop.parent / target))
                if not hop.is_relative_to(top):
                    unit = _hub_unit(root, hop)
                    if unit is None:
                        return f"{link.relative_to(top)}: link leads outside its cache", units
                    units.add(unit)
                if not hop.is_symlink():
                    break
            else:
                return f"{link.relative_to(top)}: link chain too long", units
            mode = hop.lstat().st_mode if hop.exists() else 0
            if not stat.S_ISREG(mode):
                return f"{link.relative_to(top)}: link does not end on a regular file", units
            if os.path.realpath(link) != str(hop):  # the kernel took another way (a symlinked dir mid-chain)
                return f"{link.relative_to(top)}: link chain resolves differently on disk", units
    return None, units


def hub_repo_dir(root: Path, path: Path) -> Path:
    """The unit a narrowed job mounts read-only to read ``path`` (P017 step 6, bao-security Q1), else ``path``.

    ``.cache/hf/hub/<repo>`` for a path inside an HF hub repo (its snapshot links point into ``blobs/``), and
    ``.cache/prepared/<bundle>`` for a prepared model: its ``model/`` config names sibling folders of the bundle
    (Nemotron-Parse: the RADIO encoder in ``c-radiov2-h_*``), which must be there too.
    """
    if path.is_relative_to(root):
        rel = path.relative_to(root)
        if rel.parts[:3] == (".cache", "hf", "hub") and len(rel.parts) > 3:
            return root.joinpath(*rel.parts[:4])
        if rel.parts[:2] == (".cache", "prepared") and len(rel.parts) > 2:
            return root.joinpath(*rel.parts[:3])
    return path


__all__ = [
    "DATA_DIRS",
    "HostPaths",
    "PathOutsideRootError",
    "UnsafeMountSourceError",
    "check_mount_source",
    "hub_repo_dir",
    "link_units",
    "secret_file_problem",
    "tree_problem",
]
