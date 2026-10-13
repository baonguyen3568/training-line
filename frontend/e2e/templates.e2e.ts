// Release 1.0 gate (tester, luồng F): EVERY openable template opens, renders its whole graph, and runs to the end
// (mock API, or the app on FakeWorker), in both UI skins (classic and D4, chosen with ?ui=). A template that cannot
// run must say it needs an HF account; reasons and the gallery entries that cannot be opened yet are recorded as
// annotations, so the release report lists them honestly instead of the test hiding them.
import type { Locator, Page } from "@playwright/test";
import { expect, test, tr } from "./fixtures";

interface Skin {
  name: "classic" | "d4";
  cards(page: Page): Locator; // gallery entries
  openLabel: () => string; // button that opens a template
  run(page: Page): Locator; // the Run button of the editor
}

const SKINS: Skin[] = [
  {
    name: "classic",
    cards: (page) => page.locator(".card-grid .card"),
    openLabel: () => tr("gallery.use_template"),
    run: (page) => page.getByTestId("check-bar").getByRole("button", { name: tr("app.run") }),
  },
  {
    name: "d4",
    cards: (page) => page.locator(".rp-card"),
    openLabel: () => tr("pipelines.open"),
    run: (page) => page.getByRole("button", { name: tr("app.run"), exact: true }).first(),
  },
];

for (const skin of SKINS) {
  test(`every gallery template opens and runs to the end (${skin.name})`, async ({
    page,
  }, testInfo) => {
    test.skip(testInfo.project.name === "narrow-view", "narrow screens are view-only (P010)");
    test.setTimeout(60_000 + 60_000 * 6);
    const openable = (p: Page) =>
      skin.cards(p).filter({ has: p.getByRole("button", { name: skin.openLabel() }) });

    await page.goto(`/?ui=${skin.name}`);
    await expect(skin.cards(page).first()).toBeVisible({ timeout: 10_000 });
    const count = await openable(page).count();
    expect(count).toBeGreaterThan(0);
    for (const card of await skin.cards(page).all()) {
      if ((await card.getByRole("button", { name: skin.openLabel() }).count()) === 0)
        testInfo.annotations.push({
          type: "pipeline-not-openable",
          description: (await card.innerText()).replace(/\s+/g, " ").slice(0, 160),
        });
    }

    for (let i = 0; i < count; i++) {
      await page.goto(`/?ui=${skin.name}`);
      await expect(skin.cards(page).first()).toBeVisible({ timeout: 10_000 });
      const card = openable(page).nth(i);
      const title = (await card.locator("h2").innerText()).trim();
      const needsAccount = (await card.getByTestId("needs-hf-account").count()) > 0;
      await card.getByRole("button", { name: skin.openLabel() }).click();

      await expect(page.locator(".react-flow")).toBeVisible();
      const nodes = page.locator(".react-flow__node");
      await expect(nodes.first()).toBeVisible();
      const n = await nodes.count();
      for (let k = 0; k < n; k++) await expect(nodes.nth(k)).toBeInViewport({ ratio: 0.5 }); // fitView shows all

      const run = skin.run(page);
      await expect(run).toBeVisible();
      // Validation is asynchronous: give it time to land before reading the Run state.
      await expect(run)
        .toBeEnabled({ timeout: 5_000 })
        .catch(() => undefined);
      if (!(await run.isEnabled())) {
        testInfo.annotations.push({
          type: "template-cannot-run",
          description: `${title} (needsAccount=${needsAccount})`,
        });
        expect(needsAccount, `"${title}" cannot run and does not say it needs an account`).toBe(
          true,
        );
        continue;
      }
      await run.click();
      for (let k = 0; k < n; k++) {
        await expect(nodes.nth(k).locator("[data-status]").first()).toHaveAttribute(
          "data-status",
          /^(completed|cached)$/,
          { timeout: 60_000 },
        );
      }
      testInfo.annotations.push({
        type: "template-ran",
        description: `${title}: ${n} nodes completed or cached`,
      });
    }
  });
}
