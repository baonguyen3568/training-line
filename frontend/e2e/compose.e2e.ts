// Release 1.0 luồng A: smoke of an app ALREADY RUNNING (the compose stack), run with E2E_TIER=external (see e2e.sh).
// Never presses Run on a template (that would start real GPU jobs): it checks the session, the build commit, the
// gallery, and one CPU-only job through the real worker path (model.hf_pick -> export.hf_dir), whose output is
// checked on disk (E2E_DATA_ROOT) the way a user would receive it.
import { readdirSync, rmSync } from "node:fs";
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
  const { graph_run_id: runId } = (await started.json()) as { graph_run_id: string };

  // Node statuses from the graph run's own events (the API the canvas uses), not from run.json files: a host node
  // re-run in the same run dir keeps an old run.json status while its mtime moves (backend, E2E @e75f1b9).
  // Socket auth like the app: cookie + Origin at the handshake, then {"type": "auth", csrf_token} first.
  await page.goto("/");
  const final = await page.evaluate(
    ({ id, token }) =>
      new Promise<Record<string, string>>((resolve, reject) => {
        const nodes: Record<string, string> = {};
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(
          `${proto}//${location.host}/api/runs/${encodeURIComponent(id)}/events`,
        );
        ws.onopen = () => ws.send(JSON.stringify({ type: "auth", csrf_token: token }));
        ws.onmessage = (msg) => {
          const ev = JSON.parse(String(msg.data)) as {
            type: string;
            node_id?: string;
            status?: string;
            error?: { code?: string };
          };
          if (ev.type === "node_status" && ev.node_id)
            nodes[ev.node_id] = `${ev.status}${ev.error?.code ? ` (${ev.error.code})` : ""}`;
          if (ev.type === "run_not_found") reject(new Error("run_not_found"));
          if (ev.type === "run_status" && ev.status !== "running") {
            nodes.run = String(ev.status);
            ws.close();
            resolve(nodes);
          }
        };
        ws.onclose = (e) => {
          if (!("run" in nodes)) reject(new Error(`socket closed ${e.code} before the run ended`));
        };
      }),
    { id: runId, token: csrf },
  );
  test.info().annotations.push({ type: "nodes", description: JSON.stringify(final) });
  expect(final).toEqual({
    m: expect.stringMatching(/^(completed|cached)$/),
    x: "completed",
    run: "completed",
  });
  const folder = join(dataRoot as string, dest);
  const files = readdirSync(folder);
  expect(files).toContain("config.json");
  expect(files.some((f) => f.endsWith(".safetensors"))).toBe(true);
  expect(files).toEqual(expect.arrayContaining(["LICENSE", "NOTICE", "README.md"]));
  // A test export is not kept: remove exactly this folder (under <data root>/outputs/e2e-compose/).
  rmSync(folder, { recursive: true, force: true });
});

test("QLoRA validates on Qwen3.5 and is refused where it cannot run (validate only, no job)", async ({
  page,
}) => {
  // PM 2026-10-14 (850e1ba): nf4 runs only for models whose template keeps the vision part bf16 (qlora_models).
  const session = await page.request.get("/api/session", {
    headers: { Accept: "application/json" },
  });
  expect(session.status()).toBe(200);
  const { csrf_token: csrf } = (await session.json()) as { csrf_token: string };
  const origin = new URL(process.env.E2E_BASE_URL ?? "").origin;
  const workflow = (model: string, type: string, params: Record<string, unknown>) => ({
    format: "vlm-workflow",
    schema_version: 1,
    name: `e2e-qlora-${model}`,
    nodes: [
      { id: "m", type: "model.hf_pick", version: "1.0.0", params: { model } },
      {
        id: "d",
        type: "data.hf_load",
        version: "1.0.0",
        params: { repo: "naver-clova-ix/cord-v2" },
      },
      { id: "s", type: "data.split", version: "1.0.0", params: { val_fraction: 0.1 } },
      { id: "t", type, version: "1.0.0", params: { epochs: 1, ...params } },
    ],
    edges: [
      { from: ["d", "dataset"], to: ["s", "dataset"] },
      { from: ["m", "model"], to: ["t", "base_model"] },
      { from: ["s", "train"], to: ["t", "train_data"] },
      { from: ["s", "val"], to: ["t", "val_data"] },
    ],
  });
  type Issue = { code: string; severity?: string; params?: { rule?: string } };
  const blocking = async (wf: object): Promise<Issue[]> => {
    const res = await page.request.post("/api/graphs/validate", {
      data: wf,
      headers: { "x-vlm-csrf": csrf, Origin: origin, "Content-Type": "application/json" },
    });
    expect(res.status(), await res.text()).toBe(200);
    const { errors } = (await res.json()) as { errors: Issue[] };
    return errors.filter((e) => (e.severity ?? "error") === "error"); // a not-downloaded model is only a warning
  };
  const nf4 = { quantize_base: "nf4" };
  expect(await blocking(workflow("qwen3.5-2b", "train.lora_vlm", nf4))).toEqual([]);
  expect(await blocking(workflow("qwen3.5-0.8b", "train.lora_vlm", nf4))).toEqual([]);
  // A LoRA-capable model without qlora_skip_modules: refused for nf4 only (the same graph without it is clean).
  const gemma = await blocking(workflow("gemma-4-e4b-it", "train.lora_vlm", nf4));
  expect(gemma.map((e) => [e.code, e.params?.rule])).toEqual([["CFG-005", "not_for_model"]]);
  expect(await blocking(workflow("gemma-4-e4b-it", "train.lora_vlm", {}))).toEqual([]);
  // Nemotron-Parse takes full fine-tuning only: LoRA (and so QLoRA) is refused before the nf4 check (CFG-010).
  const parse = await blocking(workflow("nemotron-parse-v1.1", "train.lora_vlm", nf4));
  expect(parse.map((e) => e.code)).toEqual(["CFG-010"]);
});
