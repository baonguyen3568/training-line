// Site QA gates (P011, owner bao-tester): G3 a11y (axe), G4 visual regression, G6 prefers-reduced-motion.
// Run via scripts/qa/site_e2e.sh (scrubbed env), not directly. Serves a STATIC build in .cache/site-e2e-dist (never
// site/dist, which other agents rebuild), so a page edited mid-run cannot change under the test.
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.SITE_E2E_PORT ?? 7881);
export const DIST = "../.cache/site-e2e-dist"; // relative to site/

export default defineConfig({
  testDir: ".",
  testMatch: /.*\.e2e\.ts$/,
  grepInvert: process.env.SITE_E2E_VISUAL ? undefined : /@visual/, // see visual.e2e.ts
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 60_000,
  reporter: [["list"]],
  outputDir: "../../.cache/site-e2e-results", // not site/test-results (untracked noise in the shared tree)
  // Visual baselines are per machine (fonts/GPU rasterization): taken on the GB10 host only.
  snapshotPathTemplate: "{testDir}/__screenshots__/{projectName}/{arg}{ext}",
  expect: { toHaveScreenshot: { maxDiffPixelRatio: 0.01, animations: "disabled", caret: "hide" } },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    browserName: "chromium",
    headless: true,
    // Same exception as the app E2E (P003, D13): no user namespaces on this host; request allowlist in fixtures.ts.
    launchOptions: { chromiumSandbox: false },
    serviceWorkers: "block",
    trace: "off",
    video: "off",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "mobile", use: { viewport: { width: 375, height: 812 } } },
    { name: "tablet", use: { viewport: { width: 768, height: 1024 } } },
    { name: "desktop", use: { viewport: { width: 1440, height: 900 } } },
  ],
  webServer: {
    // The build runs in site_e2e.sh BEFORE Playwright: the page list (fixtures.ts) is read at collection time.
    command: `python3 -m http.server ${PORT} --bind 127.0.0.1 --directory ${DIST}`,
    cwd: "..", // site/ root
    port: PORT,
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
