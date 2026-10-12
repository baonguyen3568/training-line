#!/usr/bin/env bash
# Start one team agent: run-agent.sh <role>
set -euo pipefail
ROLE="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
KICKOFF="Bạn là bao-${ROLE}. Đọc CLAUDE.md, .team/ROSTER.md, .team/BOARD.md và .team/roles/${ROLE}.md, rồi bắt đầu Phase 0 ngay (knowledge → skills → giới thiệu với đồng đội qua SendMessage → Phase 1). Báo bao-pm khi xong Phase 0."
claude -n "bao-${ROLE}" --dangerously-skip-permissions \
  --append-system-prompt "$(cat ".team/roles/${ROLE}.md")" \
  "$KICKOFF"
