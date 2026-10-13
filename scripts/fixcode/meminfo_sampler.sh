#!/usr/bin/env bash
# Read-only sampler of host unified-memory headroom (ADR-005 calibration of memguard warn/abort).
# Appends one JSON line per interval to outputs/system/meminfo-<start>.jsonl (schema vlm.system.meminfo/v1,
# units in key names, same style as runs/<id>/system.jsonl). Exits after DURATION_H hours.
# Usage: nohup setsid bash scripts/fixcode/meminfo_sampler.sh [interval_s=60] [duration_h=48] &
set -euo pipefail
INTERVAL="${1:-60}"; DURATION_H="${2:-48}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/system/meminfo-$(date +%Y%m%d-%H%M).jsonl"
END=$(( $(date +%s) + DURATION_H * 3600 ))
echo "sampler pid $$ -> $OUT" >&2
while [ "$(date +%s)" -lt "$END" ]; do
  # Lock file moved to .team/lock/GPU_LOCK; the old path still counts as held during the switch (gpu_lock guard).
  lock=false; { [ -e "$ROOT/.team/lock/GPU_LOCK" ] || [ -e "$ROOT/.team/GPU_LOCK" ]; } && lock=true
  awk -v ts="$(date -Iseconds)" -v lock="$lock" '
    /^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} /^Cached:/ {c=$2} /^SwapFree:/ {s=$2}
    END {printf "{\"schema_version\":\"vlm.system.meminfo/v1\",\"ts\":\"%s\",\"mem_total_gb\":%.2f,\"mem_available_gb\":%.2f,\"cached_gb\":%.2f,\"swap_free_gb\":%.2f,\"gpu_lock_held\":%s}\n",
         ts, t/1048576, a/1048576, c/1048576, s/1048576, lock}' /proc/meminfo >> "$OUT"
  sleep "$INTERVAL"
done
