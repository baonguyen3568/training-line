// G-3 "Sửa nhanh": the server's resolved actions (error_catalog.resolved_actions) become buttons on the error card.
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ResolvedAction, ValidationIssue } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

async function setup(actions: ResolvedAction[], over: Partial<ValidationIssue> = {}) {
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const model = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
  const train = useWorkflow.getState().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
  useWorkflow.getState().setParam(train, "local_batch_size", 8);
  const issue: ValidationIssue = {
    node_id: train,
    code: "MEM-001",
    severity: "error",
    params: { need_gb: 60, avail_gb: 40, est_gb: 22 },
    title: "Không đủ bộ nhớ",
    what: "…",
    action_ids: actions.map((a) => a.id),
    actions: actions.map((a) => ({ node_id: train, ...a })),
    ...over,
  };
  const { QuickFixes } = await import("../src/components/QuickFixes");
  const view = render(<QuickFixes issue={issue} nodeId={train} />);
  const params = () => useWorkflow.getState().workflow?.nodes.find((n) => n.id === train)?.params;
  return { useWorkflow, model, train, params, ...view };
}

const reduceBatch: ResolvedAction = {
  id: "reduce_batch",
  kind: "patch",
  confirm: false,
  patch: [{ param: "local_batch_size", from: 8, value: 4 }],
};

describe("quick fixes", () => {
  it("a patch applies the server's final value in one undo step", async () => {
    const { useWorkflow, params } = await setup([reduceBatch]);
    const toasts: string[] = [];
    window.addEventListener("vlm:toast", (e) => toasts.push((e as CustomEvent<string>).detail));
    fireEvent.click(screen.getByRole("button", { name: /Giảm batch/ }));
    expect(params()?.local_batch_size).toBe(4);
    expect(toasts[0]).toContain("Giảm batch");
    act(() => useWorkflow.getState().undo());
    expect(params()?.local_batch_size).toBe(8);
  });

  it("a stale hint (param changed since validate) shows no button", async () => {
    const { useWorkflow, train, container } = await setup([reduceBatch]);
    act(() => useWorkflow.getState().setParam(train, "local_batch_size", 2));
    expect(container.querySelector('[data-action="reduce_batch"]')).toBeNull();
  });

  it("confirm=true previews from → to before applying", async () => {
    const { params } = await setup([{ ...reduceBatch, confirm: true }]);
    fireEvent.click(screen.getByRole("button", { name: /Giảm batch/ }));
    expect(params()?.local_batch_size).toBe(8); // not yet
    // the param's label first, its id as code after it; numbers as code
    expect(screen.getByRole("group").textContent).toMatch(/\(local_batch_size\): 8 → 4/);
    fireEvent.click(screen.getByRole("button", { name: "Áp dụng" }));
    expect(params()?.local_batch_size).toBe(4);
  });

  it("graph op remove_step removes the node; unknown ids and missing placeholders get no button", async () => {
    const { useWorkflow, train, container } = await setup([
      { id: "remove_step", kind: "patch", confirm: false, patch: null },
      { id: "auto_rename_ids", kind: "patch", confirm: false, patch: null }, // no handler yet
      {
        id: "use_qlora",
        kind: "patch",
        confirm: false,
        patch: [{ param: "method", from: null, value: "qlora" }],
      },
    ]);
    expect(container.querySelector('[data-action="auto_rename_ids"]')).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Xoá bước này/ }));
    expect(useWorkflow.getState().workflow?.nodes.some((n) => n.id === train)).toBe(false);
  });

  it("navigate: node:model selects the model step, memory_panel opens Storage", async () => {
    const { useWorkflow, model } = await setup([
      { id: "smaller_model", kind: "navigate", confirm: false, target: "node:model" },
      { id: "see_memory_usage", kind: "navigate", confirm: false, target: "memory_panel" },
    ]);
    fireEvent.click(screen.getByRole("button", { name: /Chọn model nhỏ hơn/ }));
    expect(useWorkflow.getState().selectedId).toBe(model);
    const nav = vi.fn();
    window.addEventListener("vlm:navigate", (e) => nav((e as CustomEvent<string>).detail));
    fireEvent.click(screen.getByRole("button", { name: /Xem ứng dụng nào đang dùng bộ nhớ/ }));
    expect(nav).toHaveBeenCalledWith("storage");
  });

  it("node:model goes to the model feeding the failing step, not the workflow's first model", async () => {
    const { useWorkflow, train } = await setup([
      { id: "smaller_model", kind: "navigate", confirm: false, target: "node:model" },
    ]);
    const s = useWorkflow.getState();
    const other = s.addNode("model.hf_pick", { x: 0, y: 200 }) as string; // second model, wired to train
    expect(s.connect([other, "model"], [train, "base_model"]).ok).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /Chọn model nhỏ hơn/ }));
    expect(useWorkflow.getState().selectedId).toBe(other);
  });

  it("node:port selects the failing step (its picker for that port); without a port there is no button", async () => {
    const { useWorkflow, train, container } = await setup(
      [{ id: "pick_input_source", kind: "navigate", confirm: false, target: "node:port" }],
      { code: "CFG-003", port: "train_data" },
    );
    fireEvent.click(screen.getByRole("button", { name: /Chọn nguồn cho đầu vào này/ }));
    expect(useWorkflow.getState().selectedId).toBe(train);
    container.remove();
    const again = await setup([
      { id: "pick_input_source", kind: "navigate", confirm: false, target: "node:port" },
    ]);
    expect(again.container.querySelector('[data-action="pick_input_source"]')).toBeNull();
  });

  it("node:asset goes to the source step that brought the asset in", async () => {
    const { useWorkflow, train } = await setup(
      [{ id: "pick_other_asset", kind: "navigate", confirm: false, target: "node:asset" }],
      { code: "SEC-003" },
    );
    const s = useWorkflow.getState();
    let data = "";
    act(() => {
      data = s.addNode("data.hf_load", { x: 0, y: 300 }) as string;
      expect(s.connect([data, "dataset"], [train, "train_data"]).ok).toBe(true);
    });
    fireEvent.click(screen.getByRole("button", { name: /Chọn model\/dữ liệu khác/ }));
    expect(useWorkflow.getState().selectedId).toBe(data);
  });

  it("resume without a patch runs the workflow again (P017 resumes it); resume with a patch has no button", async () => {
    const { useWorkflow, container } = await setup([
      { id: "resume_checkpoint", kind: "resume", confirm: true },
      {
        id: "resume_lighter",
        kind: "resume",
        confirm: true,
        patch: [{ param: "local_batch_size", from: 8, value: 4 }],
      },
    ]);
    expect(container.querySelector('[data-action="resume_lighter"]')).toBeNull();
    const start = vi.spyOn(useWorkflow.getState(), "startRun").mockResolvedValue();
    fireEvent.click(screen.getByRole("button", { name: /Tiếp tục từ chỗ đã lưu/ }));
    expect(start).toHaveBeenCalled();
  });
});
