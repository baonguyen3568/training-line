"""Run ONE command in a project image through the job runner (preflight, GPU lock, memguard, run.json).

``uv run python -m vlm_pipeline.common.jobrun_cli --image vlm-automodel:26.08-3 --tag p2check --stage eval
    --est-peak-gb 8 -- python /workspace/scripts/qa/x.py --out RUN_DIR/pred.jsonl``

``RUN_DIR`` in the command is replaced by the run directory as seen in the container. Network none, HF offline,
non-root, repo mounted at /workspace (same baseline as every job). Exit code 0 only if the run completed.
``--read-only-project`` mounts the repo read-only; the run dir and each ``--rw-dir`` (under outputs/) stay writable.
An image bao-security has not approved yet (``--allow-unapproved-image``) ALWAYS runs read-only: its code is not
trusted to write to src/, scripts/ or .git/ (bao-security, vLLM probe). ``--unapproved-image-rw`` opts out, logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import time
from pathlib import Path

from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    ThresholdTooLooseError,
    run_job,
    tightened_thresholds,
    validate_rw_dirs,
)
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest

logger = logging.getLogger(__name__)


def build_job(a: argparse.Namespace, root: Path, run_id: str, run_dir: Path, argv_c: tuple[str, ...]) -> ContainerJob:
    """The container job for parsed ``a``: an unapproved image is read-only unless explicitly opted out."""
    read_only = a.read_only_project or (a.allow_unapproved_image and not a.unapproved_image_rw)
    if a.allow_unapproved_image and a.unapproved_image_rw:
        logger.warning("unapproved image %s runs with the project mounted READ-WRITE (--unapproved-image-rw)", a.image)
    rw_dirs = (run_dir, *(root / d for d in a.rw_dir)) if read_only else ()
    return ContainerJob(name=run_id, image=a.image, argv=argv_c, gpu=not a.no_gpu,
                        read_only_project=read_only, rw_dirs=rw_dirs, entrypoint=a.entrypoint)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.common.jobrun_cli")
    p.add_argument("--image", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--stage", default="eval", choices=["data", "train", "quantize", "eval", "bench", "serve", "export"])
    p.add_argument("--est-peak-gb", type=float, required=True)
    p.add_argument("--abort-gb", type=float)
    p.add_argument("--max-free-pct", type=float, help="use at most this %% of the free memory (< 80, the user rule)")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--who", default="cli")
    p.add_argument("--allow-unapproved-image", action="store_true",
                   help="team probes only: run on a pinned image that bao-security has not approved yet")  # fmt: skip
    p.add_argument("--read-only-project", action="store_true", help="mount the repo read-only (run dir stays rw)")
    p.add_argument("--rw-dir", action="append", default=[], type=Path, metavar="PATH",
                   help="extra writable dir under outputs/ (with a read-only project), repeatable")  # fmt: skip
    p.add_argument("--unapproved-image-rw", action="store_true",
                   help="with --allow-unapproved-image: keep the repo writable (not recommended, logged)")  # fmt: skip
    p.add_argument("--entrypoint", help="override the image ENTRYPOINT (vLLM image: the command's first word)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command ...")
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("missing command after --")
    try:  # before any run dir/manifest exists: a refused setting never leaves a "running" run behind
        thresholds = tightened_thresholds(a.abort_gb, None if a.max_free_pct is None else a.max_free_pct / 100)
    except ThresholdTooLooseError as exc:
        p.error(f"{exc} ({exc.params})")

    root = project_root()
    run_id = f"{time.strftime('%Y%m%d-%H%M')}-{a.tag}-{secrets.token_hex(2)}"
    run_dir = root / "outputs" / "runs" / run_id
    (run_dir / "logs").mkdir(parents=True)
    run_dir_c = f"{CONTAINER_ROOT}/outputs/runs/{run_id}"
    argv_c = tuple(arg.replace("RUN_DIR", run_dir_c) for arg in cmd)
    manifest = RunManifest(run_id=run_id, name=f"{a.stage} {a.tag}", stage=a.stage, task=a.stage,
                           container_image=a.image, tags=["cli"])  # fmt: skip
    write_manifest(run_dir, manifest)
    job = build_job(a, root, run_id, run_dir, argv_c)
    try:
        rw_dirs = validate_rw_dirs(root, job.rw_dirs)  # before creating anything: a rejected dir is never created
    except ValueError as exc:
        p.error(str(exc))
    for d in rw_dirs:  # docker would create a missing bind source as root
        d.mkdir(parents=True, exist_ok=True)
    final = run_job(job, manifest, run_dir, project_root=root, who=a.who, est_peak_gb=a.est_peak_gb,
                    mem_thresholds=thresholds,
                    require_approved_image=not a.allow_unapproved_image)  # fmt: skip
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "status": final.status,
                      "error_code": final.error_code}))  # fmt: skip
    return 0 if final.status == RunStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
