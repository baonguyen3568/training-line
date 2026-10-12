// E2E for the M1 journey (M1-6, owner bao-tester). Run via scripts/qa/e2e.sh (scrubbed env), not directly.
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 7876);

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
    trace: "off",
    video: "off",
    screenshot: "only-on-failure",
  },
  webServer: {
    // Tier (a): frontend with the in-browser mock API. Tier (b) (FastAPI + FakeWorker) sets E2E_SERVER_CMD.
    command:
      process.env.E2E_SERVER_CMD ??
      `VITE_MOCK_API=1 npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
    cwd: "..", // frontend/ root: vite must serve frontend/index.html, not e2e/
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
