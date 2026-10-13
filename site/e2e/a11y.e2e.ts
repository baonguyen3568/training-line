// G3 (P011): WCAG 2.2 AA with axe-core on every built page, light and dark. serious/critical violations fail;
// moderate/minor are attached to the report for design to triage.
import AxeBuilder from "@axe-core/playwright";
import { expect, sitePages, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

for (const scheme of ["light", "dark"] as const) {
  for (const path of sitePages()) {
    test(`a11y ${scheme} ${path}`, async ({ page }, testInfo) => {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(path);
      const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      const blocking = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      const summary = (vs: typeof violations) =>
        vs.map(
          (v) =>
            `${v.impact} ${v.id}: ${v.nodes.length}× ${v.nodes[0]?.target.join(" ")}` +
            ` (${v.nodes[0]?.any[0]?.message ?? v.help})`, // e.g. contrast ratio + colors
        );
      if (violations.length > blocking.length)
        await testInfo.attach("axe-non-blocking", { body: summary(violations).join("\n") });
      expect(summary(blocking)).toEqual([]);
    });
  }
}
