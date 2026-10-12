#!/usr/bin/env bash
# Read-only 1 Hz memory sampler for the shared DGX Spark (GB10 unified memory).
# nvidia-smi reports no memory on GB10, so host usage comes from /proc/meminfo.
# Usage: scripts/qa/mem_sampler.sh <out.csv> [pid]
set -euo pipefail
out=${1:?usage: mem_sampler.sh <out.csv> [pid]}
pid=${2:-}
mkdir -p "$(dirname "$out")"
echo "ts,host_used_gb,mem_avail_gb,cached_gb,swap_used_gb,pid_rss_gb" > "$out"
while true; do
  read -r used avail cached swap < <(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} /^Cached/{c=$2}
    /^SwapTotal/{st=$2} /^SwapFree/{sf=$2}
    END{printf "%.2f %.2f %.2f %.2f\n",(t-a)/1048576,a/1048576,c/1048576,(st-sf)/1048576}' /proc/meminfo)
  rss=""
  if [[ -n "$pid" ]]; then
    if [[ -r /proc/$pid/status ]]; then
      rss=$(awk '/^VmRSS/{printf "%.2f",$2/1048576}' "/proc/$pid/status")
    else
      echo "pid $pid exited" >&2; break
    fi
  fi
  echo "$(date +%s),$used,$avail,$cached,$swap,$rss" >> "$out"
  sleep 1
done
