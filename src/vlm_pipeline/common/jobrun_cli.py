"""Run ONE command in a project image through the job runner (preflight, GPU lock, memguard, run.json).

``uv run python -m vlm_pipeline.common.jobrun_cli --image vlm-automodel:26.08-3 --tag p2check --stage eval
    --est-peak-gb 8 -- python /workspace/scripts/qa/x.py --out RUN_DIR/pred.jsonl``

``RUN_DIR`` in the command is replaced by the run directory as seen in the container. Network none, HF offline,
non-root, repo mounted at /workspace (same baseline as every job). Exit code 0 only if the run completed.
"""

from __future__ import annotations

import argparse
import json
import secrets
import time

from vlm_pipeline.common.jobrun import CONTAINER_ROOT, ContainerJob, run_job, tightened_thresholds
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.common.jobrun_cli")
    p.add_argument("--image", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--stage", default="eval", choices=["data", "train", "quantize", "eval", "bench", "serve", "export"])
    p.add_argument("--est-peak-gb", type=float, required=True)
    p.add_argument("--abort-gb", type=float)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--who", default="cli")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command ...")
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("missing command after --")

    root = project_root()
    run_id = f"{time.strftime('%Y%m%d-%H%M')}-{a.tag}-{secrets.token_hex(2)}"
    run_dir = root / "outputs" / "runs" / run_id
    (run_dir / "logs").mkdir(parents=True)
    run_dir_c = f"{CONTAINER_ROOT}/outputs/runs/{run_id}"
    argv_c = tuple(arg.replace("RUN_DIR", run_dir_c) for arg in cmd)
    manifest = RunManifest(run_id=run_id, name=f"{a.stage} {a.tag}", stage=a.stage, task=a.stage,
                           container_image=a.image, tags=["cli"])  # fmt: skip
    write_manifest(run_dir, manifest)
    job = ContainerJob(name=run_id, image=a.image, argv=argv_c, gpu=not a.no_gpu)
    final = run_job(job, manifest, run_dir, project_root=root, who=a.who, est_peak_gb=a.est_peak_gb,
                    mem_thresholds=tightened_thresholds(a.abort_gb))  # fmt: skip
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "status": final.status,
                      "error_code": final.error_code}))  # fmt: skip
    return 0 if final.status == RunStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
