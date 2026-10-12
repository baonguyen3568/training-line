#!/usr/bin/env bash
# Read-only memory watch for one project container on the shared GB10 host.
# Host sampler starts immediately (baseline); once the container is running, a second sampler
# tracks the container's main PID RSS, `docker stats` is polled every 5 s and the container cgroup
# memory.current / memory.peak (includes child processes, e.g. JIT compilers) is sampled. Stops when the
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
echo "ts,cgroup_current_gb,cgroup_peak_gb" > "$out/cgroup_mem.csv"
cg=""
while (( $(date +%s) - start < max_s )); do
  state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
  if [[ "$state" == running ]]; then
    if [[ -z "$pid" ]]; then
      pid=$(docker inspect -f '{{.State.Pid}}' "$name")
      echo "$(date -Is) container running pid=$pid" >> "$out/events.log"
      "$here/mem_sampler.sh" "$out/pid_mem.csv" "$pid" & pid_sampler=$!
      cid=$(docker inspect -f '{{.Id}}' "$name")
      cg="/sys/fs/cgroup/system.slice/docker-${cid}.scope"
      [[ -r "$cg/memory.current" ]] || { echo "$(date -Is) cgroup not readable: $cg" >> "$out/events.log"; cg=""; }
    fi
    if [[ -n "$cg" && -r "$cg/memory.current" ]]; then
      cur=$(cat "$cg/memory.current"); peak=$(cat "$cg/memory.peak" 2>/dev/null || echo 0)
      awk -v t="$(date +%s)" -v c="$cur" -v p="$peak" 'BEGIN{printf "%s,%.2f,%.2f\n", t, c/1073741824, p/1073741824}' >> "$out/cgroup_mem.csv"
    fi
    docker stats --no-stream --format "$(date +%s),{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" "$name" >> "$out/docker_stats.csv" 2>/dev/null
  elif [[ -n "$pid" ]]; then
    echo "$(date -Is) container state=$state, stopping" >> "$out/events.log"
    docker inspect -f 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$name" >> "$out/events.log" 2>/dev/null
    # memory.peak is lost once the container is removed; the last sampled value above is the record.
    sleep 15  # tail after exit (memory release)
    break
  fi
  sleep 5
done
echo "$(date -Is) watch done" >> "$out/events.log"
