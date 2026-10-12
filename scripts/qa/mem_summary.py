"""Summarise a mem_sampler.sh CSV: baseline, startup peak, steady state, peak (GB). Read-only."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def summarise(path: Path, baseline_s: int = 10, steady_tail_s: int = 30) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"{path}: empty")
    used = [float(r["host_used_gb"]) for r in rows]
    avail = [float(r["mem_avail_gb"]) for r in rows]
    rss = [float(r["pid_rss_gb"]) for r in rows if r.get("pid_rss_gb")]
    baseline = statistics.median(used[:baseline_s])
    return {
        "samples": float(len(rows)),
        "baseline_used_gb": round(baseline, 2),
        "peak_used_gb": round(max(used), 2),
        "peak_delta_gb": round(max(used) - baseline, 2),
        "steady_delta_gb": round(statistics.median(used[-steady_tail_s:]) - baseline, 2),
        "min_mem_available_gb": round(min(avail), 2),
        "peak_pid_rss_gb": round(max(rss), 2) if rss else float("nan"),
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, summarise(Path(arg)))
