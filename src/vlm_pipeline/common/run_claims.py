"""One job per run dir, ever (ADR-015, bao-security M3): a claim must exist before a job gets its run dir read-write.

The app (trusted) prepares a run dir (config.yaml, request.json) BEFORE the job; the job writes in it AFTER. A claim
makes "after" happen once: a second job, a retry or a resume with the same run id is refused (``RunDirClaimedError``),
so no request can obtain write access to a folder an earlier job could have filled with symlinks or planted files.
Resume runs under a NEW run id with the old checkpoint as a read-only input.

Claims live in ``outputs/system/runner_claims/<run_id>``, which is never a job mount (rw or input), so a job cannot
delete its own claim. A claim is never removed, also not by retention: a run id is never reused. Created with
``O_CREAT|O_EXCL|O_NOFOLLOW`` through a directory fd reached by a no-symlink walk from the data root. Order for a
launcher: ``check_mount_source(run_dir)`` -> ``claim_run`` -> mount. Every launcher (runner, local docker) calls this
one function: there is no path that starts a job without a claim.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import time
from pathlib import Path

from vlm_pipeline.common.host_paths import UnsafeMountSourceError, check_mount_source

CLAIMS_REL = Path("outputs") / "system" / "runner_claims"
# A CLI/train run (``20261013-1015-p3b-full-parse-c4f6``) or a graph node run dir (``<node id>-<16 hex>``, executor),
# optionally a later attempt of it (``-a2`` .. ``-a10``, P017) and a job of that node (``-model``/``-base``: eval,
# bench).
RUN_ID = re.compile(
    r"^(?:[0-9]{8}-[0-9]{4}-[a-z0-9._-]{1,64}"
    r"|[A-Za-z0-9][A-Za-z0-9_-]{0,63}-[0-9a-f]{16}(?:-g[0-9]{1,3})?(?:-a[0-9]{1,2})?(?:-[a-z0-9_]{1,32})?)$"
)
TAG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,40}$")  # CLI --tag: keeps the CLI run id inside RUN_ID


class RunDirClaimedError(RuntimeError):
    """This run id already had a job: its run dir is never handed out read-write again."""

    reason = "run_dir_claimed"


class ClaimStoreError(RuntimeError):
    """The claims folder is unsafe (symlink, foreign owner...): no job can be claimed until it is fixed."""

    reason = "claim_store_unsafe"


def run_tag(value: str) -> str:
    """Argparse type for CLI ``--tag``: lower-case letters, digits, '.', '_', '-' (the run id stays a claim name)."""
    if not TAG.fullmatch(value):
        raise ValueError(f"tag must match {TAG.pattern}")
    return value


def claim_run(data_root: Path, run_id: str, *, by: str) -> Path:
    """Record that ``run_id`` now has its job; raise RunDirClaimedError if it ever had one. Returns the claim path."""
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("bad_run_id")
    claims = data_root
    for part in CLAIMS_REL.parts:  # one level at a time, checked before going deeper: never mkdir through a symlink
        claims = claims / part
        with contextlib.suppress(FileExistsError):  # mkdir never follows a symlink at the last component
            claims.mkdir(mode=0o700)
        try:
            check_mount_source(data_root, claims, directory=True)  # no symlink anywhere from the data root down
        except UnsafeMountSourceError as exc:
            raise ClaimStoreError(str(exc)) from None
    dir_fd = os.open(claims, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            fd = os.open(run_id, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                         dir_fd=dir_fd)  # fmt: skip
        except FileExistsError:  # also a planted symlink with that name: O_EXCL never follows it
            raise RunDirClaimedError(run_id) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": run_id, "by": by[:64], "ts": time.time()}) + "\n")
    finally:
        os.close(dir_fd)
    return claims / run_id


LEASES_REL = Path("outputs") / "system" / "resume_leases"
LATER_ATTEMPT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}-[0-9a-f]{16}(?:-g[0-9]{1,3})?-a[0-9]{1,2}$")
GENERATIONS_REL = Path("outputs") / "system" / "node_generations"
NODE_BASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}-[0-9a-f]{16}$")
MAX_GENERATION = 999


def write_resume_lease(data_root: Path, attempt: str, *, run: str, checkpoint: str, node: str) -> Path:
    """Host-side lease (P017 C6): the attempt dir ``attempt`` will resume from ``run/checkpoints/<checkpoint>``.

    Written the moment the executor picks the source, before the job queues (its run.json only says
    ``resumed_from`` once the job starts), never inside a run dir (B1). Retention keeps the source while it exists;
    the executor removes it when the node ends, ``stale_runs`` removes it when that executor is gone.
    """
    if not LATER_ATTEMPT.fullmatch(attempt):
        raise ValueError("bad_attempt")
    leases = data_root
    for part in LEASES_REL.parts:  # same no-symlink walk as the claims folder
        leases = leases / part
        with contextlib.suppress(FileExistsError):
            leases.mkdir(mode=0o700)
        try:
            check_mount_source(data_root, leases, directory=True)
        except UnsafeMountSourceError as exc:
            raise ClaimStoreError(str(exc)) from None
    body = {
        "run": run,
        "checkpoint": checkpoint,
        "node": node,
        "pid": os.getpid(),
        "ts": time.time(),
        **process_identity(),
    }
    dir_fd = os.open(leases, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(f"{attempt}.json", flags, 0o600, dir_fd=dir_fd)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(body) + "\n")
    finally:
        os.close(dir_fd)
    return leases / f"{attempt}.json"


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()[:64] or None
    except OSError:
        return None


def _pidns() -> str | None:
    try:
        return str(Path("/proc/self/ns/pid").readlink())[:64]
    except OSError:
        return None


def process_starttime(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks since boot), None when the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    fields = raw.rsplit(")", 1)[-1].split()  # after the command name, which may hold spaces or ")"
    return fields[19] if len(fields) > 19 else None  # stat field 22 = index 19 after "(comm)"


def process_identity() -> dict[str, str | None]:
    """Who wrote a lease, beyond the pid (security L1): a pid alone lies across PID namespaces and after reuse."""
    return {"boot_id": _boot_id(), "pidns": _pidns(), "starttime": process_starttime(os.getpid())}


def holder_gone(body: dict[str, object]) -> bool | None:
    """True: the process that wrote ``body`` is certainly gone; False: alive; None: cannot tell from here.

    Another boot -> gone. Same boot, another PID namespace (CLI on the host vs app in a container) -> cannot tell.
    Same boot and namespace -> gone when the pid is dead or its start time differs (the pid was reused).
    Fields missing (an older lease) -> cannot tell.
    """
    pid, boot, pidns, start = body.get("pid"), body.get("boot_id"), body.get("pidns"), body.get("starttime")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not all(isinstance(v, str) for v in (boot, pidns, start))
    ):
        return None
    local = _boot_id()
    if local is None:  # /proc hidden here (e.g. in a container): no verdict, never fail open
        return None
    if boot != local:
        return True
    if pidns != _pidns():
        return None
    return process_starttime(pid) != start


def drop_resume_lease(data_root: Path, attempt: str) -> None:
    """Remove the lease of ``attempt`` (missing is fine; never follows a symlink: unlink removes the link itself)."""
    if LATER_ATTEMPT.fullmatch(attempt):
        with contextlib.suppress(FileNotFoundError):
            (data_root / LEASES_REL / f"{attempt}.json").unlink()


class GenerationError(RuntimeError):
    """The node's generation file is unreadable or out of range: no attempt dir is guessed (fail closed)."""


def node_generation(data_root: Path, base: str) -> int:
    """Generation of ``<nid>-<key16>``: 0 without a file; "Reset this step" (Storage, bao-cleanup) bumps it (P017).

    Claims are never removed, so after a reset the node needs NEW dir names: ``<base>-g<N>[-a<M>]``.
    """
    if not NODE_BASE.fullmatch(base):
        raise GenerationError("bad_node_base")
    path = data_root / GENERATIONS_REL / f"{base}.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return 0
    except OSError as exc:  # ELOOP: a symlink
        raise GenerationError(f"generation file unusable ({exc.strerror})") from None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > 4096:
            raise GenerationError("generation file is not a small regular file")
        try:
            gen = json.loads(fh.read(4097)).get("gen")
        except (ValueError, AttributeError):
            raise GenerationError("generation file is not a JSON object") from None
    if isinstance(gen, bool) or not isinstance(gen, int) or not 0 <= gen <= MAX_GENERATION:
        raise GenerationError("generation must be an integer 0..999")
    return gen


def is_claimed(data_root: Path, run_id: str) -> bool:
    return RUN_ID.fullmatch(run_id) is not None and os.path.lexists(data_root / CLAIMS_REL / run_id)


def node_dir_used(data_root: Path, name: str) -> bool:
    """A job ran for this node run dir: its own claim, or one of its jobs' (``<name>-model``: eval/bench sub-jobs).

    A later attempt (``<name>-a2``) or generation (``<name>-g1``) is another dir, not a job of this one.
    """
    if is_claimed(data_root, name):
        return True
    claims = data_root / CLAIMS_REL
    if claims.is_symlink() or not claims.is_dir():
        return False
    job = re.compile(rf"^{re.escape(name)}-(?![ag][0-9]{{1,3}}(?:-|$))[a-z0-9_]{{1,32}}$")
    return any(job.fullmatch(entry.name) for entry in claims.iterdir())


__all__ = [
    "CLAIMS_REL",
    "RUN_ID",
    "TAG",
    "ClaimStoreError",
    "RunDirClaimedError",
    "claim_run",
    "is_claimed",
    "node_dir_used",
]
