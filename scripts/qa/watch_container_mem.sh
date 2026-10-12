#!/usr/bin/env bash
# Read-only memory watch for one project container on the shared GB10 host.
# Host sampler starts immediately (baseline); once the container is running, a second sampler
# tracks the container's main PID RSS and `docker stats` is polled every 5 s. Stops when the
# container exits (or after MAX_S seconds). Never touches the container.
# Usage: scripts/qa/watch_container_mem.sh <container_name> <out_dir> [max_s]
set -uo pipefail
name=${1:?container}; out=${2:?out_dir}; max_s=${3:-7200}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
"$here/mem_sampler.sh" "$out/host_mem.csv" & host_pid=$!
trap 'kill $host_pid ${pid_sampler:-} 2>/dev/null' EXIT
start=$(date +%s); pid=""
echo "ts,name,cpu,mem_usage,mem_pct" > "$out/docker_stats.csv"
while (( $(date +%s) - start < max_s )); do
  state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
  if [[ "$state" == running ]]; then
    if [[ -z "$pid" ]]; then
      pid=$(docker inspect -f '{{.State.Pid}}' "$name")
      echo "$(date -Is) container running pid=$pid" >> "$out/events.log"
      "$here/mem_sampler.sh" "$out/pid_mem.csv" "$pid" & pid_sampler=$!
    fi
    docker stats --no-stream --format "$(date +%s),{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" "$name" >> "$out/docker_stats.csv" 2>/dev/null
  elif [[ -n "$pid" ]]; then
    echo "$(date -Is) container state=$state, stopping" >> "$out/events.log"
    docker inspect -f 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$name" >> "$out/events.log" 2>/dev/null
    sleep 15  # tail after exit (memory release)
    break
  fi
  sleep 5
done
echo "$(date -Is) watch done" >> "$out/events.log"
