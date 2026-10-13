// E2E for the M1 journey (M1-6, owner bao-tester). Run via scripts/qa/e2e.sh (scrubbed env), not directly.
// E2E_TIER=mock (default): static build with the in-browser mock API.
// E2E_TIER=app: real FastAPI + executor + FakeWorker (scripts/qa/e2e_app.py) serving a separate build.
// Both serve a STATIC build (no HMR): other agents editing src/ in the shared tree cannot reload the page.
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 7879);
const TIER = process.env.E2E_TIER ?? "mock";
// never frontend/dist (served by the developer app); one per port, so two runs side by side never rebuild each
// other's files mid-run (2026-10-13: a second run wiped the dist a x100 run was serving)
const DIST = `../.cache/e2e-dist-${TIER}-${PORT}`;
// One results folder per run (path is relative to this file), so a later run cannot overwrite the screenshot/trace of a failure (tester 2026-10-13).
// Workers re-load this file: keep the main process's stamp (inherited env) so all artifacts land together.
process.env.E2E_STAMP ??= new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "");
const STAMP = process.env.E2E_STAMP;
const build = `npx vite build --outDir ${DIST} --emptyOutDir --logLevel warn`;
const server =
  TIER === "app"
    ? `${build} && cd .. && uv run --quiet python scripts/qa/e2e_app.py ${PORT} .cache/e2e-dist-${TIER}-${PORT}`
    : `VITE_MOCK_API=1 ${build} && npx vite preview --outDir ${DIST} --host 127.0.0.1 --port ${PORT} --strictPort`;

export default defineConfig({
  testDir: ".",
  testMatch: /.*\.e2e\.ts$/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 60_000,
  reporter: [["list"]],
  outputDir: `../../.cache/e2e-results/${TIER}-${STAMP}`,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    browserName: "chromium",
    headless: true,
    // chromiumSandbox: false — host has kernel.apparmor_restrict_unprivileged_userns=1; accepted by bao-security
    // (P003, 2026-10-12) with request allowlist + scrubbed env (see fixtures.ts and scripts/qa/e2e.sh).
    launchOptions: { chromiumSandbox: false },
    serviceWorkers: "block", // a service worker could fetch around context.route
    trace: "retain-on-failure", // only failed tests keep one; app-tier token is a per-run test token (e2e.sh)
    video: "off",
    screenshot: "only-on-failure",
  },
  // Editing is supported from 1280 px wide (P010): edit journeys run at the minimum and a wide size; below that
  // the app is view-only, so 1024 px only gets a smoke test tagged @narrow.
  projects: [
    { name: "desktop-min", use: { viewport: { width: 1280, height: 720 } }, grepInvert: /@narrow/ },
    {
      name: "desktop-wide",
      use: { viewport: { width: 1920, height: 1080 } },
      grepInvert: /@narrow/,
    },
    { name: "narrow-view", use: { viewport: { width: 1024, height: 768 } }, grep: /@narrow/ },
  ],
  webServer: {
    command: server,
    cwd: "..", // frontend/ root
    // app tier answers 401 on "/" without the session cookie; Playwright only needs the port to be up
    port: PORT,
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
