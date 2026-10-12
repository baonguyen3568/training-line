#!/usr/bin/env bash
# Read-only sampler of host unified-memory headroom (ADR-005 calibration of memguard warn/abort).
# Appends one CSV row per interval to outputs/system/meminfo-<start>.csv. Exits after DURATION_H hours.
# Usage: nohup bash scripts/fixcode/meminfo_sampler.sh [interval_s=60] [duration_h=48] &
set -euo pipefail
INTERVAL="${1:-60}"; DURATION_H="${2:-48}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/system/meminfo-$(date +%Y%m%d-%H%M).csv"
END=$(( $(date +%s) + DURATION_H * 3600 ))
echo "ts,mem_available_gb,cached_gb,swap_free_gb,gpu_lock" > "$OUT"
echo "sampler pid $$ -> $OUT" >&2
while [ "$(date +%s)" -lt "$END" ]; do
  awk -v ts="$(date -Iseconds)" -v lock="$([ -e "$ROOT/.team/GPU_LOCK" ] && echo held || echo free)" '
    /^MemAvailable:/ {a=$2} /^Cached:/ {c=$2} /^SwapFree:/ {s=$2}
    END {printf "%s,%.2f,%.2f,%.2f,%s\n", ts, a/1048576, c/1048576, s/1048576, lock}' /proc/meminfo >> "$OUT"
  sleep "$INTERVAL"
done
