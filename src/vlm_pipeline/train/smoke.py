"""Run an AutoModel recipe YAML as a smoke/calibration job (temporary CLI until the graph executor exists).

``python -m vlm_pipeline.train.smoke --config configs/train/smoke/x.yaml --tag parse-v1.1 --est-peak-gb 14
    [--prepared .cache/prepared/<dir>] [--base-model org/m --base-revision <sha>]``

Substitutes ``RUN_DIR`` / ``PREPARED_MODEL_DIR`` placeholders, runs ``automodel`` in vlm-automodel through the job
runner (preflight + GPU lock + run.json), then converts AutoModel's training.jsonl into the contract metrics.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import subprocess
import time
from pathlib import Path

from vlm_pipeline.common.jobrun import CONTAINER_ROOT, ContainerJob, run_job, tightened_thresholds
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, automodel_to_metrics, write_manifest

DEFAULT_IMAGE = "vlm-automodel:26.08-1"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.train.smoke")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tag", required=True, help="short run name, e.g. lora-parse-v1.1")
    p.add_argument("--est-peak-gb", type=float, required=True)
    p.add_argument("--ckpt-gb", type=float, default=1.0)
    p.add_argument("--prepared", type=Path, help="prepared model dir (contains model/) for PREPARED_MODEL_DIR")
    p.add_argument("--base-model")
    p.add_argument("--base-revision")
    p.add_argument("--dataset")
    p.add_argument("--task", default="lora", choices=["lora", "sft"])
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--worktree", metavar="COMMIT", help="run code from a git worktree at COMMIT (ADR-002 A1)")
    p.add_argument("--gate", action="store_true", help="refuse to run unless reproducibility_gaps() == []")
    p.add_argument("--abort-gb", type=float, help="memguard abort level; may only be HIGHER than ADR-005 (8 GB)")
    p.add_argument("--max-steps", type=int, help="override step_scheduler.max_steps (measurement runs)")
    p.add_argument("--expect-trainable-pct", metavar="LO:HI", help="fail fast if the trainable share is outside")
    return p


def main(argv: list[str] | None = None) -> int:
    p = _parser()
    a = p.parse_args(argv)

    root = project_root()
    run_id = f"{time.strftime('%Y%m%d-%H%M')}-{a.tag}-{secrets.token_hex(2)}"
    run_dir = root / "outputs" / "runs" / run_id
    (run_dir / "logs").mkdir(parents=True)
    run_dir_c = f"{CONTAINER_ROOT}/outputs/runs/{run_id}"
    code_root = None
    if a.worktree:
        code_root = root / ".worktrees" / run_id
        subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "worktree", "add", "--detach", str(code_root), a.worktree],  # noqa: S607
            check=True, capture_output=True,
        )  # fmt: skip
    text = ((code_root or root) / a.config).read_text(encoding="utf-8").replace("RUN_DIR", run_dir_c)
    if "PREPARED_MODEL_DIR" in text:
        if a.prepared is None:
            p.error("config uses PREPARED_MODEL_DIR: pass --prepared")
        prepared = json.loads((root / a.prepared / "prepared.json").read_text(encoding="utf-8"))
        text = text.replace("PREPARED_MODEL_DIR", prepared["container_dir"])
    if a.max_steps is not None:
        text = re.sub(r"^(\s*)max_steps:.*$", rf"\g<1>max_steps: {a.max_steps}", text, count=1, flags=re.M)
    (run_dir / "config.yaml").write_text(text, encoding="utf-8")

    manifest = RunManifest(
        run_id=run_id,
        name=f"smoke {a.tag}",
        stage="train",
        task=a.task,
        base_model=a.base_model,
        base_model_revision=a.base_revision,
        dataset=a.dataset,
        config_path=str(a.config),
        config_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        seed=int(m.group(1)) if (m := re.search(r"^\s*seed:\s*(\d+)", text, re.M)) else None,
        max_steps=int(m2.group(1)) if (m2 := re.search(r"^\s*max_steps:\s*(\d+)", text, re.M)) else None,
        container_image=a.image,
        tags=["smoke"],
    )
    write_manifest(run_dir, manifest)
    job = ContainerJob(
        name=run_id,
        image=a.image,
        argv=("automodel", f"{run_dir_c}/config.yaml", "--nproc-per-node", "1"),
        workdir="/opt/Automodel",
        code_root=code_root,
        env={"VLM_EXPECTED_TRAINABLE_PCT": a.expect_trainable_pct} if a.expect_trainable_pct else {},
    )
    thresholds = tightened_thresholds(a.abort_gb)
    from vlm_pipeline.common.jobrun import MemoryPreemptedBeforeStartError

    try:
        final = run_job(job, manifest, run_dir, project_root=root, who="bao-backend", est_peak_gb=a.est_peak_gb,
                        ckpt_gb=a.ckpt_gb, gate=a.gate, mem_thresholds=thresholds)  # fmt: skip
    except MemoryPreemptedBeforeStartError as exc:
        print(json.dumps({"run_id": run_id, "status": "failed", "error_code": "MEM-001", "detail": str(exc)}))
        return 2
    except RuntimeError as exc:  # preflight refusal / gate refusal
        print(json.dumps({"run_id": run_id, "status": "refused", "detail": str(exc)}))
        return 2
    for src, split in (("training.jsonl", "train"), ("validation.jsonl", "val")):
        path = run_dir / "checkpoints" / src
        if path.exists():
            lines = list(automodel_to_metrics(path.read_text(encoding="utf-8").splitlines(), split=split))  # type: ignore[arg-type]
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as out:
                out.writelines(line + "\n" for line in lines)
    print(json.dumps({"run_id": run_id, "status": final.status, "error_code": final.error_code}))
    return 0 if final.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
