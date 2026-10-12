#!/usr/bin/env bash
# Launch all team agents in tmux sessions bao-trainning-2..10 (idempotent: skips existing sessions)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROLES=(research backend tester security fixcode cleanup cleancode frontend design)
n=2
for role in "${ROLES[@]}"; do
  s="bao-trainning-$n"
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "skip $s (exists)"
  else
    tmux new-session -d -s "$s" -c "$ROOT" "bash scripts/team/run-agent.sh $role; exec bash"
    echo "started $s -> bao-$role"
    sleep 3
  fi
  n=$((n+1))
done
