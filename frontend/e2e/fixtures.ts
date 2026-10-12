// Shared E2E fixtures: request allowlist (only the app origin), abort counter asserted to be 0, i18n lookup.
import { readFileSync } from "node:fs";
import { test as base, expect } from "@playwright/test";

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

export const test = base.extend<{ blockedRequests: string[] }>({
  blockedRequests: [
    async ({ context, baseURL }, use) => {
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
      // App tier: bootstrap the session exactly like a user opening the printed URL (D13: ?token= -> cookie).
      const token = process.env.E2E_TOKEN;
      if (token) {
        const page = await context.newPage();
        await page.goto(`/?token=${token}`);
        await page.close();
      }
      await use(blocked);
      // The app must never call anything but its own origin (bao-security condition 1).
      expect(blocked, `requests outside ${allowed}`).toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
