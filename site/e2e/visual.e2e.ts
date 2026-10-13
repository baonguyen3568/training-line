// G4 (P011): visual regression, every page at 3 viewports (projects) × light/dark. Tagged @visual and run on
// demand (`SITE_E2E_VISUAL=1 scripts/qa/site_e2e.sh --grep @visual`) while M2 pages still change daily; baselines are taken on the
// GB10 host only (`--update-snapshots`), after design approves the page. [data-volatile] (dates) is masked.
import { expect, sitePages, test } from "./fixtures";

for (const scheme of ["light", "dark"] as const) {
  for (const path of sitePages()) {
    test(`visual ${scheme} ${path} @visual`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme, reducedMotion: "reduce" });
      await page.goto(path);
      await page.evaluate(() => document.fonts.ready);
      const name = `${path.replace(/^\/|\/$/g, "").replaceAll("/", "__") || "home"}-${scheme}.png`;
      await expect(page).toHaveScreenshot(name, { fullPage: true, mask: [page.locator("[data-volatile]")] });
    });
  }
}
