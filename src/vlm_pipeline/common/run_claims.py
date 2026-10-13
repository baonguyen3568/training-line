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
import time
from pathlib import Path

from vlm_pipeline.common.host_paths import UnsafeMountSourceError, check_mount_source

CLAIMS_REL = Path("outputs") / "system" / "runner_claims"
# A CLI/train run (``20261013-1015-p3b-full-parse-c4f6``) or a graph node run dir (``<node id>-<16 hex>``, executor).
RUN_ID = re.compile(r"^(?:[0-9]{8}-[0-9]{4}-[a-z0-9-]{1,48}|[A-Za-z0-9][A-Za-z0-9_-]{0,63}-[0-9a-f]{16})$")


class RunDirClaimedError(RuntimeError):
    """This run id already had a job: its run dir is never handed out read-write again."""

    reason = "run_dir_claimed"


class ClaimStoreError(RuntimeError):
    """The claims folder is unsafe (symlink, foreign owner...): no job can be claimed until it is fixed."""

    reason = "claim_store_unsafe"


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


def is_claimed(data_root: Path, run_id: str) -> bool:
    return RUN_ID.fullmatch(run_id) is not None and os.path.lexists(data_root / CLAIMS_REL / run_id)


__all__ = ["CLAIMS_REL", "RUN_ID", "ClaimStoreError", "RunDirClaimedError", "claim_run", "is_claimed"]
