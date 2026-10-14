// Shared E2E fixtures: request allowlist (only the app origin), abort counter asserted to be 0, i18n lookup.
import { readFileSync } from "node:fs";
import { test as base, type Cookie, expect } from "@playwright/test";

type Tree = { [key: string]: Tree | string };
const bundle = JSON.parse(
  readFileSync(new URL("../src/generated/i18n.vi.json", import.meta.url), "utf-8"),
) as Tree;

/** UI text for a dotted i18n key (same bundle the app renders), so tests never hardcode copy. */
export function tr(key: string): string {
  let node: Tree | string | undefined = bundle;
  for (const part of key.split(".")) {
    node = typeof node === "object" ? node[part] : undefined;
  }
  if (typeof node !== "string") throw new Error(`missing i18n key ${key}`);
  return node;
}

export const test = base.extend<{ blockedRequests: string[] }, { externalSession: Cookie[] }>({
  // External tier: sign in ONCE per worker and reuse the cookie. The edge allows 5 login POSTs per minute per IP
  // (ADR-019, nginx limit_req burst 5): one login per test ran into 429 and then the 60 s setup timeout (PM run
  // 2026-10-14 @e75f1b9). The link is /#t=<token>; the login page POSTs it to /api/session.
  externalSession: [
    async ({ browser }, use) => {
      const token = process.env.E2E_TOKEN;
      if (process.env.E2E_TIER !== "external" || !token) {
        await use([]);
        return;
      }
      const base = process.env.E2E_BASE_URL ?? "";
      const context = await browser.newContext({ baseURL: base });
      const origin = new URL(base).origin; // same allowlist as the tests: nothing but the app's origin
      await context.route("**/*", (route) =>
        new URL(route.request().url()).origin === origin
          ? route.continue()
          : route.abort("blockedbyclient"),
      );
      const page = await context.newPage();
      const login = page.waitForResponse(
        (r) => r.url().endsWith("/api/session") && r.request().method() === "POST",
        { timeout: 20_000 },
      );
      // The token must never reach a log or report: failures are thrown without the URL.
      const status = await page
        .goto(`/#t=${token}`, { timeout: 20_000 })
        .then(() => login)
        .then((r) => r.status())
        .catch(() => 0);
      const check = await page.request.get("/api/session", {
        headers: { Accept: "application/json" },
      });
      const cookies = await context.cookies();
      await context.close();
      if (status !== 200 || check.status() !== 200)
        throw new Error(
          `login failed: POST /api/session -> ${status}, GET -> ${check.status()} (URL withheld)`,
        );
      await use(cookies);
    },
    { scope: "worker" },
  ],
  blockedRequests: [
    async ({ context, baseURL, externalSession }, use) => {
      const allowed = new URL(baseURL ?? "http://127.0.0.1").origin;
      const blocked: string[] = [];
      await context.route("**/*", (route) => {
        const url = route.request().url();
        if (url.startsWith("data:") || url.startsWith("blob:") || new URL(url).origin === allowed) {
          return route.continue();
        }
        blocked.push(url);
        return route.abort("blockedbyclient");
      });
      // context.route does not see WebSockets: refuse any socket to another origin too.
      await context.routeWebSocket(
        (url) => url.origin.replace(/^ws/, "http") !== allowed,
        (ws) => {
          blocked.push(ws.url());
          return ws.close();
        },
      );
      // Sign in like a user opening the printed link. App tier (local Host): GET /?token= -> cookie (D13), per test
      // (a fresh test-only token, no rate limit). External: the worker's one session (externalSession above).
      const token = process.env.E2E_TOKEN;
      if (externalSession.length > 0) {
        await context.addCookies(externalSession);
      } else if (token && process.env.E2E_TIER !== "external") {
        const page = await context.newPage();
        const ok = await page
          .goto(`/?token=${token}`)
          .then((r) => r !== null && r.status() < 400)
          .catch(() => false);
        await page.close();
        if (!ok) throw new Error("login with E2E_TOKEN failed (URL withheld)");
      }
      await use(blocked);
      // The app must never call anything but its own origin (bao-security condition 1).
      expect(blocked, `requests outside ${allowed}`).toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
