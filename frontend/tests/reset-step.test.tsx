// G-4 "Đặt lại bước" (bao-cleanup storage_reset): plan → typed node title (accent-insensitive) → reset; blocked plans
// say why; one button per graph node in Storage; the canvas offers it only for a node that already ran.
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { NodeResetPlan } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

const plan = (over: Partial<NodeResetPlan> = {}): NodeResetPlan => ({
  schema_version: 1,
  plan_id: "p1",
  expires_at: Date.now() / 1000 + 600,
  node: "n4-0123456789abcdef",
  node_title: "Dạy model",
  attempts: [
    { run_dir: "n4-0123456789abcdef-a1", status: "failed", size_gb: 4.25, has_checkpoint: true },
  ],
  total_gb: 4.25,
  blocked_by: null,
  blocked_params: {},
  consequence_code: "PERMANENT_LOSS",
  consequence_params: { name: "Dạy model" },
  confirm_required: true,
  confirm_text: "Dạy model",
  next_generation: 2,
  ...over,
});

async function dialog(p: NodeResetPlan) {
  const { api } = await import("../src/api");
  vi.spyOn(api, "nodeResetPlan").mockResolvedValue(p);
  const reset = vi
    .spyOn(api, "nodeReset")
    .mockResolvedValue({ deleted: ["x"], failed: [], freed_gb: 4.25, generation: 2 });
  const { ResetStepDialog } = await import("../src/components/ResetStepDialog");
  const onDone = vi.fn();
  const onClose = vi.fn();
  render(<ResetStepDialog node={p.node} onClose={onClose} onDone={onDone} />);
  await screen.findByRole("alertdialog");
  return { reset, onDone, onClose };
}

describe("reset step dialog", () => {
  it("says what is lost and needs the node title typed (accents optional) before resetting", async () => {
    const { reset, onDone } = await dialog(plan());
    await screen.findByText(/xoá 1 lần chạy của bước "Dạy model" và giải phóng 4.3 GB/);
    expect(screen.getByText(/bản lưu \(checkpoint\)/)).toBeTruthy();
    expect(screen.getByText(/Nhật ký và biểu đồ/)).toBeTruthy(); // always: the reset removes logs too
    const go = screen.getByRole("button", { name: "Đặt lại bước" });
    expect(go).toHaveProperty("disabled", true);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "day  MODEL" } });
    expect(go).toHaveProperty("disabled", false);
    fireEvent.click(go);
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(reset).toHaveBeenCalledWith(
      { plan_id: "p1", node: "n4-0123456789abcdef", confirm: "day  MODEL" },
      "vi",
    );
    expect(onDone.mock.calls[0]?.[0]).toContain('Đã đặt lại bước "Dạy model"');
    // the node is named by its title, never by its run-dir id
    expect(document.body.textContent).not.toContain("n4");
  });

  it("a blocked plan explains why and offers no reset button", async () => {
    await dialog(
      plan({
        plan_id: null,
        blocked_by: "RUN_NOT_FINISHED",
        blocked_params: { node: "Dạy model" },
      }),
    );
    await waitFor(() => expect(document.querySelector(".reset-blocked")).not.toBeNull());
    expect(screen.queryByRole("button", { name: "Đặt lại bước" })).toBeNull();
    expect(document.querySelector(".reset-blocked")?.textContent ?? "").not.toMatch(
      /\{\w+\}|RUN_NOT_FINISHED/,
    );
  });

  it("a node that never ran has nothing to reset", async () => {
    await dialog(plan({ attempts: [], plan_id: null, total_gb: 0 }));
    await screen.findByText(/chưa chạy lần nào/);
  });

  it("an expired plan is fetched again and the user confirms on the fresh one", async () => {
    const { ApiError } = await import("../src/api/client");
    const { api } = await import("../src/api");
    const { reset } = await dialog(plan({ confirm_required: false, confirm_text: null }));
    reset.mockRejectedValueOnce(new ApiError(409, { code: "PLAN_EXPIRED" }));
    const fetches = vi.mocked(api.nodeResetPlan).mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Đặt lại bước" }));
    await screen.findByText(/Danh sách đã cũ/);
    expect(vi.mocked(api.nodeResetPlan).mock.calls.length).toBe(fetches + 1);
  });
});

describe("entry points", () => {
  it("Storage shows one reset button per graph node", async () => {
    const { StoragePanel } = await import("../src/components/StoragePanel");
    render(<StoragePanel />);
    await screen.findByRole("heading", { name: "Dung lượng" });
    expect(screen.getAllByRole("button", { name: /Đặt lại bước/ })).toHaveLength(2); // n4 (2 items) + n5
  });

  it("the inspector offers it only once the node has a run-dir base", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { Inspector } = await import("../src/components/Inspector");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    useWorkflow.getState().select(id);
    render(<Inspector />);
    expect(screen.queryByRole("button", { name: /Đặt lại bước/ })).toBeNull();
    act(() =>
      useWorkflow.setState({
        run: {
          id: "r",
          done: true,
          startedAt: 0,
          nodes: { [id]: { status: "failed", nodeBase: `${id}-0123456789abcdef` } },
        },
      }),
    );
    const { useUi } = await import("../src/store/ui");
    fireEvent.click(screen.getByRole("button", { name: /Đặt lại bước/ }));
    expect(useUi.getState().resetNode).toBe(`${id}-0123456789abcdef`);
  });

  it("the store keeps node_base from node_status events", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { setFakeStepMs } = await import("../src/mocks/api");
    setFakeStepMs(5);
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    await useWorkflow.getState().startRun();
    await waitFor(() => expect(useWorkflow.getState().run?.done).toBe(true));
    expect(useWorkflow.getState().run?.nodes[id]?.nodeBase).toBe(`${id}-0123456789abcdef`);
  });
});
