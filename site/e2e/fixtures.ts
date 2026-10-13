// Every page test runs behind a request allowlist: only the local static server. Any other request (CDN, font,
// tracker) is aborted AND fails the test, so "works offline / no third party" is checked on every page. page.route
// does not see WebSockets (service workers are blocked in the config), so any socket opened fails the test too.
import { readdirSync, statSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { test as base, expect } from "@playwright/test";

declare global {
  interface Window {
    __csp?: string[];
  }
}

export const test = base.extend<{ external: string[]; pageHealth: string[] }>({
  external: [
    async ({ page, baseURL }, use) => {
      const external: string[] = [];
      const origin = new URL(baseURL ?? "http://127.0.0.1").origin;
      page.on("websocket", (ws) => external.push(ws.url()));
      await page.route("**/*", (route) => {
        const url = route.request().url();
        if (url.startsWith("data:") || new URL(url).origin === origin) return route.continue();
        external.push(url);
        return route.abort("blockedbyclient");
      });
      await use(external);
      expect(external, "requests outside the local site").toEqual([]);
    },
    { auto: true },
  ],
  // CSP violations (the policy is a <meta>, so it applies on the static server too), uncaught JS errors and console
  // errors fail the page: axe cannot see an inline script blocked by CSP or a sidebar broken by a blocked style="".
  pageHealth: [
    async ({ page }, use) => {
      const problems: string[] = [];
      page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
      page.on("console", (msg) => {
        if (msg.type() === "error") problems.push(`console: ${msg.text()}`);
      });
      await page.addInitScript(() =>
        document.addEventListener("securitypolicyviolation", (e) => {
          window.__csp ??= [];
          window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`);
        }),
      );
      await use(problems);
      if (page.url().startsWith("http")) {
        const csp = await page.evaluate(() => window.__csp ?? []);
        problems.push(...csp.map((c) => `csp: ${c}`));
      }
      expect(problems, "CSP violations / JS errors on the page").toEqual([]);
    },
    { auto: true },
  ],
});
export { expect };

/** Every built page as a URL path ("/docs/bat-dau/"), read from the static build the server serves. */
export function sitePages(): string[] {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..", ".cache", "site-e2e-dist");
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (name === "index.html") out.push(`/${relative(root, dir).split(sep).join("/")}/`.replace("//", "/"));
      else if (name === "404.html" && dir === root) out.push("/404.html"); // error page: users land here too
    }
  };
  walk(root);
  return out.sort();
}
