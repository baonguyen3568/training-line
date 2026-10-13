// Release 1.0 (D1: D4 is the default UI): the M1 acceptance journey in the D4 shell — blank workflow -> add 3 nodes
// from the library flyout -> wire (drag + inspector) -> Run -> every node completed -> save -> reopen. Plus a node
// dragged from the flyout lands in its own (compact, empty) lane. Classic keeps m1-journey.e2e.ts (?ui=classic).
import { readFileSync } from "node:fs";
import type { Locator, Page } from "@playwright/test";
import { expect, test, tr } from "./fixtures";

const NODES = ["model.hf_pick", "data.local_folder", "train.sft_vlm"] as const;

const node = (page: Page, typeId: string): Locator => page.getByTestId(`node-${typeId}`);

async function nodeId(page: Page, typeId: string): Promise<string> {
  const wrapper = page.locator(".react-flow__node").filter({ has: node(page, typeId) });
  const id = await wrapper.getAttribute("data-id");
  if (!id) throw new Error(`node ${typeId} has no data-id`);
  return id;
}

async function waitUntilStill(locator: Locator): Promise<void> {
  let previous = "";
  await expect
    .poll(
      async () => {
        const box = JSON.stringify(await locator.boundingBox());
        const still = box === previous;
        previous = box;
        return still;
      },
      { intervals: [150], timeout: 5_000 },
    )
    .toBe(true);
}

async function openLibrary(page: Page): Promise<Locator> {
  const flyout = page.locator("#library-flyout");
  if (await flyout.isHidden()) await page.locator("#strip-library").click();
  await expect(flyout).toBeVisible();
  return flyout;
}

async function addNode(page: Page, typeId: string): Promise<void> {
  const flyout = await openLibrary(page);
  await flyout.getByRole("searchbox").fill(typeId.split(".").pop() ?? typeId);
  await flyout.getByTestId(`library-item-${typeId}`).dblclick();
  await expect(node(page, typeId)).toBeInViewport();
}

async function blankWorkflow(page: Page): Promise<void> {
  await page.goto("/?ui=d4");
  await page
    .getByRole("button", { name: tr("gallery.blank_workflow") })
    .first()
    .click();
  await expect(page.locator(".react-flow")).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(0);
}

test("D4 journey: build, run, save, reopen", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === "narrow-view", "narrow screens are view-only (P010)");
  await blankWorkflow(page);
  for (const t of NODES) await addNode(page, t);
  await expect(page.locator(".react-flow__node")).toHaveCount(3);

  const [model, data, train] = await Promise.all(NODES.map((t) => nodeId(page, t)));
  const source = page.locator(
    `[data-id="${model}"] .react-flow__handle.source[data-handleid="model"]`,
  );
  const target = page.locator(
    `[data-id="${train}"] .react-flow__handle.target[data-handleid="base_model"]`,
  );
  await waitUntilStill(source);
  await waitUntilStill(target);
  const a = await source.boundingBox();
  const b = await target.boundingBox();
  if (!a || !b) throw new Error("handle not visible");
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await page.mouse.up();
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
  await node(page, NODES[2]).click();
  await page.locator("#recv-train_data").selectOption(`${data}:dataset`);
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);

  await expect(page.getByText(tr("app.demo_mode"))).toBeVisible();
  const run = page.getByRole("button", { name: tr("app.run"), exact: true }).first();
  await expect(run).toBeEnabled();
  await run.click();
  for (const t of NODES) {
    await expect(node(page, t)).toHaveAttribute("data-status", /^(completed|cached)$/, {
      timeout: 30_000,
    });
  }

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: tr("workflow.save") }).click(),
  ]);
  const saved = testInfo.outputPath("saved-d4.vlmflow.json");
  await download.saveAs(saved);
  const wf = JSON.parse(readFileSync(saved, "utf-8")) as {
    nodes: { type: string }[];
    edges: unknown[];
  };
  expect(wf.nodes.map((n) => n.type).sort()).toEqual([...NODES].sort());
  expect(wf.edges).toHaveLength(2);

  await page
    .getByRole("button", { name: tr("gallery.blank_workflow") })
    .first()
    .click();
  await expect(page.locator(".react-flow__node")).toHaveCount(0);
  await page.getByTestId("workflow-file-input").setInputFiles(saved);
  await expect(page.locator(".react-flow__node")).toHaveCount(3);
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
});

test("a node dragged from the library lands in its own lane, without a snap notice", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name === "narrow-view", "narrow screens are view-only (P010)");
  await blankWorkflow(page);
  const flyout = await openLibrary(page);
  await flyout.getByRole("searchbox").fill("modelopt");
  await flyout
    .getByTestId("library-item-quantize.modelopt_ptq")
    .dragTo(page.locator(".react-flow__pane"), { targetPosition: { x: 400, y: 250 } });
  await expect(node(page, "quantize.modelopt_ptq")).toBeInViewport();
  // The notice has {node}/{lane} placeholders: match its fixed words, else the check could never fail.
  const snapped = tr("library.snapped_to_lane")
    .split(/\{[^}]*\}/)
    .map((part) => part.trim())
    .filter((part) => part.length > 3);
  expect(snapped.length).toBeGreaterThan(0);
  for (const part of snapped) await expect(page.getByText(part, { exact: false })).toHaveCount(0);
});
