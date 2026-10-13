#!/usr/bin/env bash
# Run the frontend E2E (Playwright) with a scrubbed environment (bao-security conditions, P003, 2026-10-12):
# no HF/NGC/W&B tokens, no .env, temporary browser profile, only 127.0.0.1.
# Usage: [E2E_TIER=mock|app] scripts/qa/e2e.sh [playwright args]   (app tier: a fresh test-only token per run)
#        E2E_TIER=external E2E_BASE_URL=http://127.0.0.1:<port> E2E_LOGIN_URL='<printed link>' \
#          [E2E_DATA_ROOT=<data root>] [E2E_EXPECT_COMMIT=<sha>] scripts/qa/e2e.sh
#        an already running app (compose): only compose.e2e.ts; the login URL is never printed or stored (PM 2026-10-13)
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
token=""
if [ "${E2E_TIER:-mock}" = app ]; then
  token=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
elif [ "${E2E_TIER:-mock}" = external ]; then
  # The token is read from the login URL here and handed to the browser only; never echoed.
  token=$(E2E_LOGIN_URL="${E2E_LOGIN_URL:-}" python3 -c 'import os,urllib.parse as u
q=u.parse_qs(u.urlsplit(os.environ["E2E_LOGIN_URL"]).query).get("token",[""])[0]
print(q)')
  [ -n "$token" ] || { echo "E2E_LOGIN_URL has no token=... (URL not shown)" >&2; exit 2; }
fi
cd "$root/frontend"
exec env -i \
  PATH="$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" \
  PLAYWRIGHT_BROWSERS_PATH="$root/.cache/ms-playwright" \
  ${E2E_PORT:+E2E_PORT="$E2E_PORT"} E2E_TIER="${E2E_TIER:-mock}" \
  E2E_TOKEN="$token" ${E2E_BASE_URL:+E2E_BASE_URL="$E2E_BASE_URL"} \
  ${E2E_DATA_ROOT:+E2E_DATA_ROOT="$E2E_DATA_ROOT"} ${E2E_EXPECT_COMMIT:+E2E_EXPECT_COMMIT="$E2E_EXPECT_COMMIT"} \
  npx playwright test --config e2e/playwright.config.ts "$@"
