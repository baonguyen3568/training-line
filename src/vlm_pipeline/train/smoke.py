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
import sys
import time
from pathlib import Path

from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.jobrun import CONTAINER_ROOT, ContainerJob, run_job, tightened_thresholds
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, automodel_to_metrics, read_manifest, write_manifest


def _complete_checkpoints(run_dir: Path, base_dir: Path, root: Path) -> None:
    """BUG-002: every consolidated export (save_consolidated every|final) gets the tensors consolidation dropped.

    Runs on the host with the CURRENT checkout, so run.json records which code completed the checkpoints (ADR-002 A1).
    Only ``src/`` (+ the uv env files) is used by this step, so only those count as dirty. Re-running it
    (``smoke complete``) replaces the earlier entry of the same step; the replaced entries are kept whole in
    ``supersedes`` (the step is idempotent: a re-run restores nothing, the first run's ``restored`` lists stay there).
    """
    from vlm_pipeline.common.jobrun import git_state
    from vlm_pipeline.train.ckpt_complete import complete_from_base

    commit, dirty = git_state(root, ("src", "pyproject.toml", "uv.lock"))
    steps = []
    for consolidated in sorted((run_dir / "checkpoints").glob("epoch_*_step_*/model/consolidated")):
        restored = complete_from_base(base_dir, consolidated)
        steps.append({"checkpoint": str(consolidated.relative_to(run_dir)), "restored": restored})
        print(json.dumps({"completed_from_base": steps[-1]["checkpoint"], "restored": restored}))
    manifest = read_manifest(run_dir)
    post: dict[str, object] = {"name": "complete_from_base", "git_commit": commit, "git_dirty": dirty,
                               "checkpoints": steps}  # fmt: skip
    kept = [s for s in manifest.post_steps if s.get("name") != post["name"]]
    if replaced := [s for s in manifest.post_steps if s.get("name") == post["name"]]:
        post["supersedes"] = replaced
    write_manifest(run_dir, manifest.model_copy(update={"post_steps": [*kept, post]}))


def _complete_main(argv: list[str]) -> int:
    """``smoke complete RUN_DIR --prepared DIR``: (re)run the host post-step of a completed run from this checkout."""
    p = argparse.ArgumentParser(prog="python -m vlm_pipeline.train.smoke complete")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--prepared", type=Path, required=True)
    a = p.parse_args(argv)
    root = project_root()
    run_dir = a.run_dir if a.run_dir.is_absolute() else root / a.run_dir
    if read_manifest(run_dir).status != RunStatus.COMPLETED:
        p.error("run is not completed")
    _complete_checkpoints(run_dir, root / a.prepared / "model", root)
    return 0


def _with_restore_from(text: str, ckpt_c: str) -> str:
    """Set ``checkpoint.restore_from`` (AutoModel resumes model + optimizer + dataloader state from that dir)."""
    if re.search(r"^\s+restore_from:", text, re.M):
        raise ValueError("config already sets checkpoint.restore_from")
    out, n = re.subn(r"^checkpoint:\n", f"checkpoint:\n  restore_from: {ckpt_c}\n", text, count=1, flags=re.M)
    if n != 1:
        raise ValueError("config has no top-level checkpoint: block")
    return out


_CKPT_PARTS = ("model", "optim", "step_scheduler.pt")


_RESUME_MAY_CHANGE = frozenset({"lr", "weight_decay"})  # optimizer keys that do not change the saved state layout


def _optimizer_block(text: str) -> dict[str, object] | None:
    """Parsed top-level ``optimizer:`` block, None when the recipe has none.

    Parsed, not string-compared: the checkpoint's config.yaml is AutoModel's re-dump (``1.0e-05``, block lists).
    """
    import yaml

    block = (yaml.safe_load(text) or {}).get("optimizer")
    return {k: _as_number(v) for k, v in block.items()} if isinstance(block, dict) else None


def _as_number(value: object) -> object:
    """YAML 1.1 reads ``1e-8`` (no dot) as a string but AutoModel dumps ``1.0e-08``: compare numbers as floats."""
    if isinstance(value, list):
        return [_as_number(v) for v in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _check_restore_source(ckpt: Path, text: str) -> str | None:
    """Why ``ckpt`` cannot be resumed by ``text`` (fail before the GPU lock, not after loading the model), or None.

    The optimizer must match the parent's except ``_RESUME_MAY_CHANGE``: e.g. FusedAdam master_weights false -> true
    would find no master/remainder state and silently rebuild it from the bf16 params.
    """
    if missing := [part for part in _CKPT_PARTS if not (ckpt / part).exists()]:
        return f"{ckpt} is not a training checkpoint (missing {', '.join(missing)})"
    parent_cfg = ckpt / "config.yaml"
    if not parent_cfg.is_file():
        return f"{ckpt} has no config.yaml: cannot verify that its optimizer state matches the recipe"
    parent, child = _optimizer_block(parent_cfg.read_text(encoding="utf-8")), _optimizer_block(text)
    if parent is None or child is None:
        if parent is child:  # both use the AutoModel default optimizer
            return None
        return f"optimizer block only in the {'recipe' if parent is None else 'checkpoint'}"
    diff = sorted(k for k in (parent.keys() | child.keys()) - _RESUME_MAY_CHANGE if parent.get(k) != child.get(k))
    if diff:
        return "optimizer differs from the checkpoint's: " + ", ".join(
            f"{k} {parent.get(k)} -> {child.get(k)}" for k in diff
        )
    return None


def _render_config(a: argparse.Namespace, p: argparse.ArgumentParser, root: Path, code: Path, run_dir_c: str) -> str:
    """The recipe as the container sees it: placeholders substituted, CLI overrides applied."""
    text: str = (code / a.config).read_text(encoding="utf-8").replace("RUN_DIR", run_dir_c)
    if "PREPARED_MODEL_DIR" in text:
        if a.prepared is None:
            p.error("config uses PREPARED_MODEL_DIR: pass --prepared")
        prepared = json.loads((root / a.prepared / "prepared.json").read_text(encoding="utf-8"))
        text = text.replace("PREPARED_MODEL_DIR", str(prepared["container_dir"]))
    if a.max_steps is not None:
        text = re.sub(r"^(\s*)max_steps:.*$", rf"\g<1>max_steps: {a.max_steps}", text, count=1, flags=re.M)
    if a.restore_from is not None:
        ckpt = (root / a.restore_from).resolve()
        if not ckpt.is_dir() or not ckpt.is_relative_to(root / "outputs" / "runs"):
            p.error("--restore-from must be an existing checkpoint dir under outputs/runs/")
        if problem := _check_restore_source(ckpt, text):
            p.error(problem)
        try:
            text = _with_restore_from(text, f"{CONTAINER_ROOT}/{ckpt.relative_to(root).as_posix()}")
        except ValueError as exc:
            p.error(str(exc))
    return text


def _restored_tags(root: Path, restore_from: Path | None) -> list[str]:
    """``resumed`` + ``restored_from:<run_id>/<ckpt>``: the result depends on the parent run (see its run.json)."""
    if restore_from is None:
        return []
    rel = (root / restore_from).resolve().relative_to(root / "outputs" / "runs")
    return ["resumed", f"restored_from:{rel.parts[0]}/{rel.parts[-1]}"]


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
    p.add_argument("--image", default=DEFAULT_TRAIN_IMAGE)
    p.add_argument("--worktree", metavar="COMMIT", help="run code from a git worktree at COMMIT (ADR-002 A1)")
    p.add_argument("--gate", action="store_true", help="refuse to run unless reproducibility_gaps() == []")
    p.add_argument("--abort-gb", type=float, help="memguard abort level; may only be HIGHER than ADR-005 (8 GB)")
    p.add_argument("--max-steps", type=int, help="override step_scheduler.max_steps (measurement runs)")
    p.add_argument(
        "--restore-from",
        type=Path,
        metavar="CKPT_DIR",
        help="resume from a checkpoint dir of an earlier run (outputs/runs/<run>/checkpoints/epoch_*_step_*)",
    )
    p.add_argument("--expect-trainable-pct", metavar="LO:HI", help="fail fast if the trainable share is outside")
    return p


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["complete"]:
        return _complete_main(args[1:])
    p = _parser()
    a = p.parse_args(args)

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
    text = _render_config(a, p, root, code_root or root, run_dir_c)
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
        tags=["smoke", *_restored_tags(root, a.restore_from)],
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
    except RuntimeError as exc:  # preflight / gate refusal: nothing ran -> discard the run, drop the worktree
        refused = read_manifest(run_dir)
        if not refused.status.is_terminal:
            write_manifest(run_dir, refused.model_copy(update={"status": RunStatus.DISCARDED, "ended_at": time.time(),
                                                                "tags": [*refused.tags, "refused"]}))  # fmt: skip
        if code_root is not None:
            subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(code_root)],  # noqa: S603,S607
                           check=False, capture_output=True)  # fmt: skip
        print(json.dumps({"run_id": run_id, "status": "refused", "detail": str(exc)}))
        return 2
    if a.prepared is not None and final.status == "completed":
        _complete_checkpoints(run_dir, root / a.prepared / "model", root)
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
