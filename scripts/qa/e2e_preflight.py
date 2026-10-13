"""Memory preflight of the browser E2E (scripts/qa/e2e.sh): E2E may run while a GPU job holds the lock (PM 2026-10-13).

The browser tiers take no GPU lock (mock API, or the app on FakeWorker), so the only risk is pushing a running GPU job
into its memguard abort. Allowed when host MemAvailable >= max(16 GB, abort level of the job holding the lock + 4 GB);
the abort level is ``memguard_abort_gb`` of that job's run.json (resolved by the watchdog, ADR-005 A1). Lock read via
``gpu_lock status`` (never the lock file by hand). Prints one line; exit 0 = go, 3 = not enough memory, with the
numbers.   uv run python scripts/qa/e2e_preflight.py
Known gap: a GPU job that starts after this check is not seen; acceptable because the suite runs ~1-2 minutes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vlm_pipeline.common.memguard import read_meminfo
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.run_claims import RUN_ID
from vlm_pipeline.common.runinfo import read_manifest

FLOOR_GB = 16.0
MARGIN_OVER_ABORT_GB = 4.0
EXIT_NO_MEM = 3


def required_gb(holder_abort_gb: float | None) -> float:
    if holder_abort_gb is None:
        return FLOOR_GB
    return max(FLOOR_GB, holder_abort_gb + MARGIN_OVER_ABORT_GB)


def worst_case_abort_gb(mem_total_gb: float) -> float:
    """Highest abort level a job can have: (1 - max_free_frac) x its start-time free memory, at most MemTotal."""
    from vlm_pipeline.common.memguard import MemThresholds

    return MemThresholds().for_start(mem_total_gb).abort_gb


# Where a GPU job's run.json lives: CLI jobs (jobrun_cli, smoke), graph node runs, and the container jobs of a graph
# node (eval / bench / trial: <node run>/jobs/<name>, run_id "<node run>-<name>").
DIRECT_RUN_DIRS = ("runs", "graph/runs")
NODE_JOB_RUN_JSONS = "graph/runs/*/jobs/*/run.json"


def holder_run_dir(run_id: str, outputs: Path) -> Path | None:
    """Run dir of the lock holder's job under ``outputs``, matched on the run.json's own run_id; None if not found."""
    for sub in DIRECT_RUN_DIRS:
        run_dir = outputs / sub / run_id
        if (run_dir / "run.json").is_file():
            return run_dir
    for path in outputs.glob(NODE_JOB_RUN_JSONS):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("run_id") == run_id:
                return path.parent
        except (OSError, ValueError, AttributeError):
            continue
    return None


def holder_abort_gb(status: dict[str, object], outputs: Path) -> tuple[float | None, str]:
    """Abort level of the lock holder's job, and who holds it ("" when free); None when the run does not record it."""
    if not status.get("held"):
        return None, ""
    who = f"{status.get('who')} ({status.get('job')})"
    run_id = status.get("run_id")
    run_dir = holder_run_dir(run_id, outputs) if isinstance(run_id, str) and RUN_ID.fullmatch(run_id) else None
    if run_dir is None:
        return None, who
    level = getattr(read_manifest(run_dir), "memguard_abort_gb", None)
    return (float(level) if level is not None else None), who


def main() -> int:
    root = project_root()
    out = subprocess.run(
        [sys.executable, "-m", "vlm_pipeline.common.gpu_lock", "status"], capture_output=True, text=True, check=False
    ).stdout
    status = json.loads(out or "{}")
    abort, who = holder_abort_gb(status, root / "outputs")
    mem = read_meminfo()
    assumed = ""
    if who and abort is None:  # held, level not recorded (yet): assume the highest it can be, never the floor
        abort = worst_case_abort_gb(mem.mem_total_gb)
        assumed = " (assumed worst case: the job has not recorded it)"
    need = required_gb(abort)
    avail = mem.mem_available_gb
    holder = f"GPU lock held by {who}, its abort level {abort:.1f} GB{assumed}" if who else "GPU lock free"
    verdict = "OK" if avail >= need else "NOT ENOUGH MEMORY"
    print(f"e2e preflight: {verdict}: MemAvailable {avail:.1f} GB, need {need:.1f} GB ({holder})")
    return 0 if avail >= need else EXIT_NO_MEM


if __name__ == "__main__":
    sys.exit(main())
