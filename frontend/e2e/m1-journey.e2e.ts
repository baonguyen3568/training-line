// M1 acceptance journey (BOARD M1-6): blank workflow -> drag 3 nodes -> connect -> Run (FakeWorker / mock)
// -> see node status -> save -> reopen. Selectors agreed with bao-frontend (36b3081); UI text via i18n keys.
import { readFileSync } from "node:fs";
import type { Locator, Page } from "@playwright/test";
import { expect, test, tr } from "./fixtures";

const NODES = ["model.hf_pick", "data.local_folder", "train.sft_vlm"] as const;

function node(page: Page, typeId: string): Locator {
  return page.getByTestId(`node-${typeId}`);
}

async function nodeId(page: Page, typeId: string): Promise<string> {
  // data-testid sits inside the React Flow wrapper, which carries data-id
  const wrapper = page.locator(".react-flow__node").filter({ has: node(page, typeId) });
  const id = await wrapper.getAttribute("data-id");
  if (!id) throw new Error(`node ${typeId} has no data-id`);
  return id;
}

/** Resolves when the element's box stops moving (two equal reads 150 ms apart). */
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

/** Resolves when the element's text stops changing (two equal reads 300 ms apart). */
async function waitUntilSettled(locator: Locator): Promise<void> {
  let previous = "";
  await expect
    .poll(
      async () => {
        const text = await locator.innerText();
        const still = text === previous;
        previous = text;
        return still;
      },
      { intervals: [300], timeout: 10_000 },
    )
    .toBe(true);
}

// Forward the app's "[wire] connected from the DOM drop target" fallback log to the runner, so a x100 run can
// count how often React Flow still misses a port (should be 0 since nodes carry `measured`).
const forwarding = new WeakSet<Page>();
async function dragHandle(page: Page, from: Locator, to: Locator): Promise<void> {
  if (!forwarding.has(page)) {
    forwarding.add(page);
    page.on("console", (m) => {
      if (m.text().startsWith("[wire]")) console.log(`${m.text()} (${test.info().title})`);
    });
  }
  const a = await from.boundingBox();
  const b = await to.boundingBox();
  if (!a || !b) throw new Error("handle not visible");
  const x = a.x + a.width / 2;
  const y = a.y + a.height / 2;
  await page.mouse.move(x, y);
  // Diagnosis for the rare "drag pans the canvas" flake (tester x100, 2026-10-13): what is under the pointer
  // right before mouse.down. React Flow handles only take the pointer with class connectionindicator.
  const under = await page.evaluate(
    ([px, py]) => {
      const el = document.elementFromPoint(px, py);
      const h = el?.closest(".react-flow__handle");
      return {
        el: `${el?.tagName}.${el?.className}`,
        node: el?.closest("[data-id]")?.getAttribute("data-id") ?? null,
        handle: h ? h.className : null,
        narrow: window.matchMedia("(max-width: 1279px)").matches,
        width: `${window.innerWidth}/${document.documentElement.clientWidth}`,
      };
    },
    [x, y] as const,
  );
  if (!under.handle?.includes("connectionindicator")) {
    console.log(`[drag-diag] pointer not on a connectable handle: ${JSON.stringify(under)}`);
    test.info().annotations.push({ type: "drag-diag", description: JSON.stringify(under) });
  }
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await page.mouse.up();
}

async function addFromLibrary(page: Page, typeId: string, how: "drag" | "dblclick"): Promise<void> {
  const search = page.getByRole("searchbox", { name: tr("library.search") });
  await search.fill(typeId.split(".").pop() ?? typeId);
  const item = page.getByTestId(`library-item-${typeId}`);
  if (how === "drag") {
    await item.dragTo(page.locator(".react-flow__pane"), { targetPosition: { x: 300, y: 200 } });
  } else {
    await item.dblclick();
  }
  await search.fill("");
  // toBeVisible passes for a node outside the canvas viewport; the user must actually see the node they added.
  await expect(node(page, typeId)).toBeInViewport();
}

async function buildGraph(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(0);

  await addFromLibrary(page, NODES[0], "drag"); // real HTML5 drag-and-drop onto the canvas
  await addFromLibrary(page, NODES[1], "dblclick");
  await addFromLibrary(page, NODES[2], "dblclick");
  await expect(page.locator(".react-flow__node")).toHaveCount(3);

  const [model, data, train] = await Promise.all(NODES.map((t) => nodeId(page, t)));
  // Wire 1 by dragging handle to handle on the canvas, once any fitView animation has settled (a user drags
  // a still canvas; measuring mid-animation would aim at a moving target).
  const source = page.locator(
    `[data-id="${model}"] .react-flow__handle.source[data-handleid="model"]`,
  );
  const target = page.locator(
    `[data-id="${train}"] .react-flow__handle.target[data-handleid="base_model"]`,
  );
  await waitUntilStill(source); // the first node dropped is the likeliest to still be moving (narrow fitView)
  await waitUntilStill(target);
  await waitUntilSettled(page.getByTestId("check-bar")); // validation re-render must not land mid-drag
  await dragHandle(page, source, target);
  // Assert the drag itself, so a miss (or a refused connection) fails here, not at the inspector wire below.
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
  // Wire 2 through the inspector ("Nhận từ — <port>"), the keyboard-accessible path.
  await node(page, NODES[2]).click();
  await page.locator("#recv-train_data").selectOption(`${data}:dataset`);
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  await expect(page.locator("#recv-base_model")).toHaveValue(`${model}:model`);
}

test("first screen is the gallery and offers a blank workflow", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: tr("gallery.title") })).toBeVisible();
  await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  await expect(page.locator(".react-flow")).toBeVisible();
});

test("M1 journey: build, run, save, reopen", async ({ page }, testInfo) => {
  const appTier = process.env.E2E_TIER === "app";
  const sockets: string[] = [];
  page.on("websocket", (ws) => sockets.push(ws.url()));
  await buildGraph(page);
  // Both tiers show the demo banner (mock API, or a backend on FakeWorker): never mistaken for real training.
  await expect(page.getByText(tr("app.demo_mode"))).toBeVisible();

  // Run: every node ends completed (or cached); the check bar shows no blocking issue.
  await expect(page.getByTestId("check-bar").locator('[data-code][class*="error"]')).toHaveCount(0);
  const checkBar = page.getByTestId("check-bar");
  const run = checkBar.getByRole("button", { name: tr("app.run") });
  await expect(run).toBeEnabled();
  await run.click();
  for (const t of NODES) {
    await expect(node(page, t)).toHaveAttribute("data-status", /^(completed|cached)$/, {
      timeout: 30_000,
    });
  }
  await expect(checkBar.getByRole("button", { name: tr("app.run") })).toBeEnabled();
  // Proof of tier: the app tier streams run events over the backend WebSocket; the mock opens none.
  const runSockets = sockets.filter((u) => /\/api\/runs\/[^/]+\/events/.test(u));
  expect(runSockets.length > 0).toBe(appTier);

  // Save: a plain download of the workflow file.
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: tr("workflow.save") }).click(),
  ]);
  const saved = testInfo.outputPath("saved.vlmflow.json");
  await download.saveAs(saved);
  const wf = JSON.parse(readFileSync(saved, "utf-8")) as {
    nodes: { id: string; type: string }[];
    edges: { from: string[]; to: string[] }[];
  };
  expect(wf.nodes.map((n) => n.type).sort()).toEqual([...NODES].sort());
  expect(wf.edges).toHaveLength(2);

  // Reopen: start from an empty workflow, open the saved file, the same graph comes back.
  await page
    .getByRole("button", { name: tr("gallery.blank_workflow") })
    .first()
    .click();
  await expect(page.locator(".react-flow__node")).toHaveCount(0);
  await page.getByTestId("workflow-file-input").setInputFiles(saved);
  await expect(page.locator(".react-flow__node")).toHaveCount(3);
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
  for (const t of NODES) await expect(node(page, t)).toBeVisible();
});

test("work survives a page reload (autosave)", async ({ page }) => {
  // Regression guard for autosave (bao-design M1 review r1, K1). UX approved by design: after a reload the
  // app asks to restore the draft (draft.restore_prompt) and "Khôi phục" reopens the canvas.
  await buildGraph(page);
  // Wait for the debounced autosave to actually land (a fixed sleep is flaky on a busy host).
  await expect
    .poll(
      () =>
        page.evaluate(() => {
          const raw = localStorage.getItem("vlm.draft.v1"); // src/lib/draft.ts KEY
          // src/lib/draft.ts stores { savedAt, workflow: <workflow file as a JSON string> }
          const wf = raw ? JSON.parse(JSON.parse(raw).workflow) : null;
          return `${wf?.nodes?.length ?? 0} nodes, ${wf?.edges?.length ?? 0} edges`;
        }),
      { timeout: 10_000 },
    )
    .toBe("3 nodes, 2 edges"); // the LAST edit is saved, not an earlier draft
  await page.reload();
  await page.getByRole("button", { name: tr("draft.restore") }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(3, { timeout: 5_000 });
  await expect(page.locator(".react-flow__edge")).toHaveCount(2);
});

test("gallery lists templates without an error @narrow", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: tr("gallery.use_template") }).first(),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("narrow screens can open a template and see the canvas @narrow", async ({ page }) => {
  // Regression guard: < 1280 px is view-only (P010) but the canvas must stay visible (fixed 2026-10-12).
  await page.goto("/");
  const useTemplate = page.getByRole("button", { name: tr("gallery.use_template") }).first();
  await expect(useTemplate).toBeVisible({ timeout: 10_000 }); // fail fast: an expected failure must not time out
  await useTemplate.click();
  await expect(page.locator(".react-flow")).toBeVisible();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await expect(page.getByTestId("check-bar")).toBeVisible();
});

test("handles do not move when validation results arrive (no layout shift)", async ({ page }) => {
  // Root cause of a lost-wire flake (bao-frontend, 2026-10-12): validation inserted a hint inside the node and
  // thickened its border, pushing handles down mid-drag. Guard: the target handle keeps its position.
  await page.goto("/");
  await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  await addFromLibrary(page, NODES[2], "dblclick");
  const handle = page.locator(".react-flow__handle.target[data-handleid='base_model']");
  const before = await handle.boundingBox();
  // Validation (CFG-003: inputs missing) has landed on the node itself: the handle is marked needs-wire, the
  // render that used to shift it. ([data-code] in the check bar only renders once "details" is expanded.)
  await expect(handle).toHaveClass(/\bneeds-wire\b/);
  const after = await handle.boundingBox();
  // <= 0.5 px tolerates subpixel rounding; a real shift was ~7 px (half a hint line).
  expect(Math.abs((after?.y ?? 0) - (before?.y ?? 0))).toBeLessThanOrEqual(0.5);
  expect(Math.abs((after?.x ?? 0) - (before?.x ?? 0))).toBeLessThanOrEqual(0.5);
});

test("a wire dragged right after adding the nodes connects (no waiting)", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: tr("gallery.blank_workflow") }).click();
  await addFromLibrary(page, NODES[0], "dblclick");
  await addFromLibrary(page, NODES[2], "dblclick");
  const [model, train] = await Promise.all([nodeId(page, NODES[0]), nodeId(page, NODES[2])]);
  await dragHandle(
    page,
    page.locator(`[data-id="${model}"] .react-flow__handle.source[data-handleid="model"]`),
    page.locator(`[data-id="${train}"] .react-flow__handle.target[data-handleid="base_model"]`),
  );
  await expect(page.locator(".react-flow__edge")).toHaveCount(1);
});
