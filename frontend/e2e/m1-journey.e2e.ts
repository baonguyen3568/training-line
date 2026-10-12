// M1 acceptance journey (BOARD M1-6): blank workflow -> drag 3 nodes -> connect -> Run (FakeWorker)
// -> see node status/results -> save -> reopen. Steps that need M1-4 UI are added as it lands.
import { expect, test, tr } from "./fixtures";

test("first screen is the gallery and offers a blank workflow", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: tr("gallery.title") })).toBeVisible();
  await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
});
