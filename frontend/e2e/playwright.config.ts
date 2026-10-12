// E2E for the M1 journey (M1-6, owner bao-tester). Run via scripts/qa/e2e.sh (scrubbed env), not directly.
// E2E_TIER=mock (default): static build with the in-browser mock API.
// E2E_TIER=app: real FastAPI + executor + FakeWorker (scripts/qa/e2e_app.py) serving a separate build.
// Both serve a STATIC build (no HMR): other agents editing src/ in the shared tree cannot reload the page.
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 7879);
const TIER = process.env.E2E_TIER ?? "mock";
const DIST = `../.cache/e2e-dist-${TIER}`; // never frontend/dist (served by the developer app)
const build = `npx vite build --outDir ${DIST} --emptyOutDir --logLevel warn`;
const server =
  TIER === "app"
    ? `${build} && cd .. && uv run --quiet python scripts/qa/e2e_app.py ${PORT} .cache/e2e-dist-app`
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
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    browserName: "chromium",
    headless: true,
    // chromiumSandbox: false — host has kernel.apparmor_restrict_unprivileged_userns=1; accepted by bao-security
    // (P003, 2026-10-12) with request allowlist + scrubbed env (see fixtures.ts and scripts/qa/e2e.sh).
    launchOptions: { chromiumSandbox: false },
    serviceWorkers: "block", // a service worker could fetch around context.route
    trace: "off",
    video: "off",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: server,
    cwd: "..", // frontend/ root
    // app tier answers 401 on "/" without the session cookie; Playwright only needs the port to be up
    port: PORT,
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
