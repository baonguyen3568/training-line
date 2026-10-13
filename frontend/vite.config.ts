/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server binds localhost only (P006 §8), in the project's port range 4000–4050 (D-8: 4010–4019 for
// dev servers). The FastAPI app serves the production build; in dev, /api is proxied to the dev app
// (VLM_DEV_APP, default 127.0.0.1:4003 per D-8) unless VITE_MOCK_API=1.
// (process.env read directly, never loadEnv: that would read .env files, which agents must not)
const env =
  (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const DEV_APP = env.VLM_DEV_APP ?? "http://127.0.0.1:4003";
export default defineConfig({
  plugins: [react()],
  // Relative asset URLs: the same build works at / and behind nginx under a prefix (e.g. /app/), D-8.
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 4010,
    strictPort: true,
    proxy: {
      "/api": { target: DEV_APP, changeOrigin: false, ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    // No CDN at runtime: everything is bundled.
    assetsInlineLimit: 0,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}", "tests/**/*.test.{ts,tsx}"],
    // Shared host: a GPU job next door can push jsdom component tests past 5 s (13:34, load 8); 15 s is still
    // far below a real hang.
    testTimeout: 15_000,
  },
});
