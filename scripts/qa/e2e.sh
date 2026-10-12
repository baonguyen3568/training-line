#!/usr/bin/env bash
# Run the frontend E2E (Playwright) with a scrubbed environment (bao-security conditions, P003, 2026-10-12):
# no HF/NGC/W&B tokens, no .env, temporary browser profile, only 127.0.0.1.
# Usage: [E2E_TIER=mock|app] scripts/qa/e2e.sh [playwright args]   (app tier: a fresh test-only token per run)
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root/frontend"
exec env -i \
  PATH="$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
  PLAYWRIGHT_BROWSERS_PATH="$root/.cache/ms-playwright" \
  ${E2E_PORT:+E2E_PORT="$E2E_PORT"} E2E_TIER="${E2E_TIER:-mock}" \
  E2E_TOKEN="$( [ "${E2E_TIER:-mock}" = app ] && python3 -c 'import secrets;print(secrets.token_urlsafe(32))' )" \
  npx playwright test --config e2e/playwright.config.ts "$@"
