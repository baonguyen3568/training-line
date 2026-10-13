#!/usr/bin/env bash
# Site QA gates (P011 G3/G4/G6) with a scrubbed environment, like scripts/qa/e2e.sh: no tokens, no .env,
# only 127.0.0.1. Builds the site into .cache/site-e2e-dist first (the page list is read from it), then runs
# Playwright on that static copy.   Usage: [SITE_E2E_VISUAL=1] scripts/qa/site_e2e.sh [playwright args]
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root/site"
clean_env=(env -i PATH="$PATH" HOME="$HOME" LANG="${LANG:-C.UTF-8}" ASTRO_TELEMETRY_DISABLED=1)
"${clean_env[@]}" npm run --silent tokens
"${clean_env[@]}" npx astro build --outDir "$root/.cache/site-e2e-dist" --silent
exec "${clean_env[@]}" PLAYWRIGHT_BROWSERS_PATH="$root/.cache/ms-playwright" \
  ${SITE_E2E_PORT:+SITE_E2E_PORT="$SITE_E2E_PORT"} ${SITE_E2E_VISUAL:+SITE_E2E_VISUAL=1} \
  npx playwright test --config e2e/playwright.config.ts "$@"
