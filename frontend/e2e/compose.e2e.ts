// Release 1.0 luồng A: smoke of an app ALREADY RUNNING (the compose stack), run with E2E_TIER=external (see e2e.sh).
// Never presses Run on a template (that would start real GPU jobs): it checks the session, the build commit, the
// gallery, and one CPU-only job through the real worker path (model.hf_pick -> export.hf_dir), whose output is
// checked on disk (E2E_DATA_ROOT) the way a user would receive it.
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, test, tr } from "./fixtures";

test.skip(process.env.E2E_TIER !== "external", "only against a running app (E2E_TIER=external)");

test("the running app answers with the expected build, signed in", async ({ page }) => {
  const res = await page.request.get("/api/health");
  expect(res.status()).toBe(200);
  const health = (await res.json()) as {
    ok: boolean;
    commit: string | null;
    dirty: boolean | null;
  };
  expect(health.ok).toBe(true);
  expect(health.dirty).toBe(false);
  const expected = process.env.E2E_EXPECT_COMMIT;
  if (expected) expect(health.commit ?? "").toMatch(new RegExp(`^${expected}`));
  test.info().annotations.push({
    type: "build",
    description: `commit ${health.commit}`,
  });
});

test("the gallery lists the templates and opens one without running it", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(tr("app.demo_mode"))).toHaveCount(0); // real worker, never the demo banner
  const open = page.getByRole("button", { name: tr("pipelines.open") });
  await expect(open.first()).toBeVisible({ timeout: 15_000 });
  await open.first().click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
});

test("G-5: a catalog model downloads through the app (egress proxy), then is checked", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-wide", "one download is enough (runs once)");
  test.setTimeout(45 * 60_000);
  const session = await page.request.get("/api/session", {
    headers: { Accept: "application/json" },
  });
  expect(session.status()).toBe(200);
  const { csrf_token: csrf } = (await session.json()) as { csrf_token: string };
  const origin = new URL(process.env.E2E_BASE_URL ?? "").origin; // the server checks Origin on writes
  const headers = {
    "x-vlm-csrf": csrf,
    Origin: origin,
    "Content-Type": "application/json",
  };
  const body = { kind: "model", catalog_id: "qwen3.5-0.8b" };
  let started = await page.request.post("/api/downloads", {
    data: body,
    headers,
  });
  expect(started.status(), await started.text()).toBe(202);
  let reply = (await started.json()) as {
    download_id: string | null;
    status?: string;
    verified?: boolean;
    needs_confirm?: boolean;
  };
  if (reply.needs_confirm) {
    started = await page.request.post("/api/downloads", {
      data: { ...body, confirm: true },
      headers,
    });
    reply = (await started.json()) as typeof reply;
  }
  if (reply.status === "completed" && reply.download_id === null && reply.verified !== true) {
    // already in the cache but never checked by the app: "Kiểm tra" re-sends with check
    started = await page.request.post("/api/downloads", {
      data: { ...body, check: true },
      headers,
    });
    reply = (await started.json()) as typeof reply;
  }
  test.info().annotations.push({ type: "download", description: JSON.stringify(reply) });
  if (reply.download_id === null) {
    expect(reply.status).toBe("completed");
    return;
  }
  const id = reply.download_id;
  const state = async (): Promise<string> => {
    const list = await page.request.get("/api/downloads", {
      headers: { Accept: "application/json" },
    });
    const rows = (await list.json()) as {
      download_id: string;
      status: string;
      error?: unknown;
    }[];
    const row = rows.find((r) => r.download_id === id);
    return row ? `${row.status}${row.error ? ` ${JSON.stringify(row.error)}` : ""}` : "missing";
  };
  await expect
    .poll(state, { intervals: [5_000], timeout: 44 * 60_000 })
    .toMatch(/^(completed|failed|cancelled)/);
  expect(await state()).toBe("completed");
});

test("a CPU job runs end to end through the real worker: pick a model, export it", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-wide", "one export is enough (runs once)");
  const dataRoot = process.env.E2E_DATA_ROOT;
  test.skip(!dataRoot, "set E2E_DATA_ROOT to check the export on disk");
  // Needs Qwen3.5-0.8B in the stack's cache: the G-5 test above downloads it (or install.sh prefetch); otherwise the
  // pick node fails NET-001, which this test then reports as the failure it is.
  test.setTimeout(15 * 60_000);
  const dest = `outputs/e2e-compose/export-${Date.now()}`;
  const session = await page.request.get("/api/session", {
    headers: { Accept: "application/json" },
  });
  expect(session.status()).toBe(200);
  const { csrf_token: csrf } = (await session.json()) as { csrf_token: string };
  const workflow = {
    format: "vlm-workflow",
    schema_version: 1,
    name: "e2e-compose-export",
    nodes: [
      {
        id: "m",
        type: "model.hf_pick",
        version: "1.0.0",
        params: { model: "qwen3.5-0.8b" },
      },
      { id: "x", type: "export.hf_dir", version: "1.0.0", params: { dest } },
    ],
    edges: [{ from: ["m", "model"], to: ["x", "model"] }],
  };
  const origin = new URL(process.env.E2E_BASE_URL ?? "").origin; // the server checks Origin on writes
  const started = await page.request.post("/api/runs", {
    data: workflow,
    headers: {
      "x-vlm-csrf": csrf,
      Origin: origin,
      "Content-Type": "application/json",
    },
  });
  expect(started.status(), await started.text()).toBe(202);

  // Status of the export node from its run.json (node runs live in outputs/graph/runs/<node>-<key>[-aN]): fail fast
  // with the error code when the node fails, and check the folder only once the node says completed.
  const graphRuns = join(dataRoot as string, "outputs", "graph", "runs");
  const since = Date.now() - 5_000;
  // Node run dirs written since the POST: "m-" (pick) and "x-" (export). A failed pick skips the export, so any
  // failed node ends the wait at once.
  const nodeStatus = (prefix: string): { status: string; error_code?: string | null } | null => {
    if (!existsSync(graphRuns)) return null;
    const dirs = readdirSync(graphRuns)
      .filter((d) => d.startsWith(prefix) && statSync(join(graphRuns, d)).mtimeMs >= since)
      .filter((d) => existsSync(join(graphRuns, d, "run.json")))
      .sort((a, b) => statSync(join(graphRuns, b)).mtimeMs - statSync(join(graphRuns, a)).mtimeMs);
    return dirs.length
      ? JSON.parse(readFileSync(join(graphRuns, dirs[0], "run.json"), "utf-8"))
      : null;
  };
  const ended = (): string => {
    const pick = nodeStatus("m-");
    if (pick && pick.status !== "completed" && pick.status !== "running")
      return `pick ${pick.status} (${pick.error_code ?? "no code"})`;
    const exp = nodeStatus("x-");
    if (exp && exp.status !== "running")
      return `export ${exp.status} (${exp.error_code ?? "no code"})`;
    return "pending";
  };
  await expect.poll(ended, { intervals: [3_000], timeout: 14 * 60_000 }).not.toBe("pending");
  expect(ended()).toBe("export completed (no code)");
  const folder = join(dataRoot as string, dest);
  const files = readdirSync(folder);
  expect(files).toContain("config.json");
  expect(files.some((f) => f.endsWith(".safetensors"))).toBe(true);
  expect(files).toEqual(expect.arrayContaining(["LICENSE", "NOTICE", "README.md"]));
  // A test export is not kept: remove exactly this folder (under <data root>/outputs/e2e-compose/).
  rmSync(folder, { recursive: true, force: true });
});
