#!/usr/bin/env bash
# Run the frontend E2E (Playwright) with a scrubbed environment (bao-security conditions, P003, 2026-10-12):
# no HF/NGC/W&B tokens, no .env, temporary browser profile, only 127.0.0.1.
# Usage: [E2E_TIER=mock|app] scripts/qa/e2e.sh [playwright args]   (app tier: a fresh test-only token per run)
# Runs while a GPU job holds the lock (no tier takes it: mock API, or the app on FakeWorker) when enough memory is
# free for the browser without pushing that job into its memguard abort: scripts/qa/e2e_preflight.py (PM 2026-10-13).
# The app tier refuses to start on anything but FakeWorker (scripts/qa/e2e_app.py).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
(cd "$root" && uv run --quiet python scripts/qa/e2e_preflight.py) || exit $?
# Each run writes .cache/e2e-results/<tier>-<stamp> (playwright.config.ts): keep the 20 newest, inside that dir only.
results="$root/.cache/e2e-results"
if [ -d "$results" ]; then
  find "$results" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -rn | tail -n +21 | cut -d' ' -f2- |
    xargs -r -d '\n' rm -rf --
fi
cd "$root/frontend"
exec env -i \
  PATH="$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
  PLAYWRIGHT_BROWSERS_PATH="$root/.cache/ms-playwright" \
  ${E2E_PORT:+E2E_PORT="$E2E_PORT"} E2E_TIER="${E2E_TIER:-mock}" \
  E2E_TOKEN="$( [ "${E2E_TIER:-mock}" = app ] && python3 -c 'import secrets;print(secrets.token_urlsafe(32))' )" \
  npx playwright test --config e2e/playwright.config.ts "$@"
