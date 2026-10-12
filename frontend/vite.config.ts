/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server binds localhost only (P006 §8). The FastAPI app (127.0.0.1:7870) serves the
// production build; in dev, /api is proxied to it unless VITE_MOCK_API=1.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 7875,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:7870", changeOrigin: false, ws: true },
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
  },
});
