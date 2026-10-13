// M1 journey on the mock API: blank workflow → add nodes → connect → undo/redo → save/open → run (FakeWorker).
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

async function setup() {
  const { useWorkflow } = await import("../src/store/workflow");
  const { setFakeStepMs } = await import("../src/mocks/api");
  setFakeStepMs(5);
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  return useWorkflow;
}

describe("M1 journey (store)", () => {
  it("adds, connects, undoes and redoes", async () => {
    const wfStore = await setup();
    const s = () => wfStore.getState();
    const model = s().addNode("model.hf_pick", { x: 0, y: 0 });
    const data = s().addNode("data.hf_load", { x: 0, y: 200 });
    const train = s().addNode("train.sft_vlm", { x: 500, y: 0 });
    expect(model && data && train).toBeTruthy();
    expect(s().workflow?.nodes.find((n) => n.id === model)?.params).toEqual({
      model: "nemotron-parse-v1.1",
    });
    expect(s().connect([model as string, "model"], [train as string, "base_model"]).ok).toBe(true);
    // Type check: a Dataset cannot feed the base_model (Model) port.
    expect(s().connect([data as string, "dataset"], [train as string, "base_model"])).toMatchObject(
      { ok: false, expected: "Model", actual: "Dataset" },
    );
    expect(s().workflow?.edges).toHaveLength(1);
    s().undo();
    expect(s().workflow?.edges).toHaveLength(0);
    s().redo();
    expect(s().workflow?.edges).toHaveLength(1);
  });

  it("duplicates without wires and deletes with its wires", async () => {
    const wfStore = await setup();
    const s = () => wfStore.getState();
    const m = s().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const tr = s().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
    s().connect([m, "model"], [tr, "base_model"]);
    const copy = s().duplicateNode(tr) as string;
    expect(s().workflow?.edges.some((e) => e.to[0] === copy)).toBe(false);
    s().removeNodes([m]);
    expect(s().workflow?.edges).toHaveLength(0);
    expect(s().workflow?.nodes.map((n) => n.id)).toEqual([tr, copy]);
  });

  it("round-trips through the workflow file", async () => {
    const wfStore = await setup();
    const { parseWorkflowFile, serializeWorkflow } = await import("../src/lib/workflowFile");
    const s = () => wfStore.getState();
    const m = s().addNode("model.hf_pick", { x: 10, y: 20 }) as string;
    const tr = s().addNode("train.sft_vlm", { x: 300, y: 20 }) as string;
    s().connect([m, "model"], [tr, "base_model"]);
    const before = s().workflow;
    const parsed = parseWorkflowFile(
      serializeWorkflow(
        before ??
          (() => {
            throw new Error();
          })(),
      ),
    );
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      s().openWorkflow(parsed.workflow);
      expect(s().workflow?.nodes).toEqual(before?.nodes);
      expect(s().workflow?.edges).toEqual(before?.edges);
    }
  });

  it("runs on the FakeWorker and reports every node done", async () => {
    const wfStore = await setup();
    const s = () => wfStore.getState();
    const m = s().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const d = s().addNode("data.hf_load", { x: 0, y: 200 }) as string;
    const tr = s().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
    s().connect([m, "model"], [tr, "base_model"]);
    s().connect([d, "dataset"], [tr, "train_data"]);
    await s().startRun();
    expect(s().run?.done).toBe(false);
    await waitFor(() => expect(s().run?.done).toBe(true), { timeout: 3000 });
    expect(Object.values(s().run?.nodes ?? {}).map((n) => n.status)).toEqual([
      "completed",
      "completed",
      "completed",
    ]);
  });
});

// Classic chrome (still reachable with ?ui=classic in 1.0); the D4 default has its own journey below.
async function classic() {
  const { useUi } = await import("../src/store/ui");
  useUi.setState({ skin: "classic" });
}

describe("M1 journey (UI)", () => {
  it("double-click in the library adds a node; Run shows per-node status", async () => {
    await classic();
    const wfStore = await setup();
    const { Editor } = await import("../src/components/Editor");
    const { container } = render(<Editor />);
    expect(screen.getByText(/Kéo "Dữ liệu" vào đây/)).toBeTruthy(); // empty canvas hint
    fireEvent.doubleClick(screen.getByTestId("library-item-model.hf_pick"));
    await waitFor(() =>
      expect(container.querySelector('[data-testid="node-model.hf_pick"]')).toBeTruthy(),
    );
    expect(wfStore.getState().workflow?.nodes).toHaveLength(1);
    await waitFor(
      () =>
        expect(screen.getByRole("button", { name: /▶ Chạy/ })).toHaveProperty("disabled", false),
      {
        timeout: 3000,
      },
    );
    fireEvent.click(screen.getByRole("button", { name: /▶ Chạy/ }));
    await waitFor(
      () =>
        expect(
          container
            .querySelector('[data-testid="node-model.hf_pick"]')
            ?.getAttribute("data-status"),
        ).toBe("completed"),
      { timeout: 3000 },
    );
  });
});

describe("Inspector enum handling", () => {
  it("keeps integer enums numeric, hides rnd/baseline models in Simple and shows RAM fit", async () => {
    const wfStore = await setup();
    const { useSystem } = await import("../src/store/system");
    useSystem.getState().set({
      mem_total_gb: 121,
      mem_available_gb: 30,
      gpu_lock: { held: false },
      disk_free_gb: 1000,
    });
    const { Inspector } = await import("../src/components/Inspector");
    const m = wfStore.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    wfStore.getState().select(m);
    const view = render(<Inspector />);
    const sel = view.container.querySelector("select") as HTMLSelectElement;
    const texts = [...sel.options].map((o) => o.textContent ?? "");
    expect(texts.some((x) => x.includes("Nemotron Parse 2.0"))).toBe(false); // rnd hidden in Simple
    expect(texts.some((x) => x.includes("Qwen3-VL 2B"))).toBe(false); // baseline hidden in Simple
    expect(texts.find((x) => x.includes("Cosmos"))).toMatch(/Hugging Face/); // gated badge
    expect(texts.find((x) => x.includes("Cosmos Reason2 8B"))).toMatch(/42 GB.*30 GB/); // 32 + 10 headroom > 30 avail
    expect(texts.find((x) => x.includes("Cosmos Reason2 2B"))).toMatch(/Vừa với/); // 14 + 10 ≤ 30
    view.unmount();
    const tr = wfStore.getState().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
    wfStore.getState().select(tr);
    const v2 = render(<Inspector />);
    const epochs = v2.container.querySelector("#param-epochs") as HTMLInputElement | null;
    expect(epochs).toBeTruthy(); // integer input (not an enum in the mock)
  });
});

describe("M1 journey (UI, D4 default)", () => {
  it("adds a node from the library flyout and Run shows per-node status", async () => {
    const wfStore = await setup();
    const { useUi } = await import("../src/store/ui");
    expect(useUi.getState().skin).toBe("d4"); // Release 1.0 default
    const { Editor } = await import("../src/components/Editor");
    const { container } = render(<Editor />);
    fireEvent.click(screen.getByRole("button", { name: "Thư viện bước" }));
    fireEvent.doubleClick(await screen.findByTestId("library-item-model.hf_pick"));
    await waitFor(() =>
      expect(container.querySelector('[data-testid="node-model.hf_pick"]')).toBeTruthy(),
    );
    expect(wfStore.getState().workflow?.nodes).toHaveLength(1);
    const run = () => screen.getByRole("button", { name: /^Chạy/ });
    await waitFor(() => expect(run()).toHaveProperty("disabled", false));
    fireEvent.click(run());
    await waitFor(() =>
      expect(
        container.querySelector('[data-testid="node-model.hf_pick"]')?.getAttribute("data-status"),
      ).toBe("completed"),
    );
  });
});

describe("cycle and empty-canvas guards (design review M1 K2/K3)", () => {
  it("rejects wiring a node from its own downstream and hides such sources in 'Nhận từ'", async () => {
    const wfStore = await setup();
    const s = () => wfStore.getState();
    const a = s().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    const b = s().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
    expect(s().connect([a, "model"], [b, "base_model"]).ok).toBe(true);
    expect(s().connect([b, "model"], [a, "base_model"]).ok).toBe(false); // would be a cycle
    expect(s().connect([a, "model"], [a, "base_model"]).ok).toBe(false); // self
    s().select(a);
    const { Inspector } = await import("../src/components/Inspector");
    const view = render(<Inspector />);
    const labels = [...view.container.querySelectorAll("option")].map((o) =>
      o.getAttribute("value"),
    );
    expect(labels.some((v) => v?.startsWith(`${a}:`) || v?.startsWith(`${b}:`))).toBe(false);
  });

  it("shows 'add a step' instead of 'ready' on an empty canvas", async () => {
    await classic();
    await setup();
    const { Editor } = await import("../src/components/Editor");
    render(<Editor />);
    expect(screen.getByText(/Thêm bước để bắt đầu/)).toBeTruthy();
    expect(screen.queryByText(/Mọi thứ đã sẵn sàng/)).toBeNull();
  });
});

describe("guidance on unconnected ports and errors in the inspector (design review M1 r1)", () => {
  it("hints the missing required port on the node and lists the error in the inspector", async () => {
    const wfStore = await setup();
    const tr = wfStore.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    await wfStore.getState().validate();
    wfStore.getState().select(tr);
    const { Editor } = await import("../src/components/Editor");
    const view = render(<Editor />);
    await waitFor(() => expect(view.container.querySelectorAll(".port.needs-wire").length).toBe(2)); // base_model + train_data
    expect(view.container.querySelector(".port-hint")?.textContent).toMatch(/nối/i);
    const cards = view.container.querySelectorAll(".inspector .error-card[data-code='CFG-003']");
    expect(cards.length).toBe(2);
    fireEvent.click(cards[0]?.querySelector("button") as HTMLElement);
    expect(document.activeElement?.id).toMatch(/^recv-/);
  });
});
