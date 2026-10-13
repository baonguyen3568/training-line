// D5 "Chờ bộ nhớ trống" (executor node_state waiting_memory/waiting_gpu): the node says it waits, with the numbers in
// its tooltip; the next node_status clears it.
import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RunEvent } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

describe("waiting for memory", () => {
  it("shows the waiting state with need/avail and clears on the next status", async () => {
    const { api } = await import("../src/api");
    let emit: (ev: RunEvent) => void = () => {};
    vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "r1" });
    vi.spyOn(api, "watchRun").mockImplementation((_id, onEvent) => {
      emit = onEvent;
      return () => {};
    });
    const { useWorkflow } = await import("../src/store/workflow");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    await useWorkflow.getState().startRun();
    const { container } = render(<Editor />);
    act(() => emit({ type: "node_status", node_id: id, status: "running" }));
    act(() =>
      emit({
        type: "node_state",
        node_id: id,
        state: "waiting_memory",
        need_gb: 41.6,
        avail_gb: 30.2,
      }),
    );
    const badge = container.querySelector('[data-waiting="waiting_memory"]');
    expect(badge?.textContent).toContain("Chờ bộ nhớ trống");
    expect(badge?.getAttribute("title")).toContain("42 GB");
    expect(badge?.getAttribute("title")).toContain("30 GB");
    act(() => emit({ type: "node_status", node_id: id, status: "running", progress: 0.1 }));
    expect(container.querySelector("[data-waiting]")).toBeNull();
    expect(screen.getAllByText(/10/).length).toBeGreaterThan(0);
  });

  it("a CUDA OOM at load retried once says so (reason gpu_alloc_transient), never the raw reason", async () => {
    const { api } = await import("../src/api");
    let emit: (ev: RunEvent) => void = () => {};
    vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "r2" });
    vi.spyOn(api, "watchRun").mockImplementation((_id, onEvent) => {
      emit = onEvent;
      return () => {};
    });
    const { useWorkflow } = await import("../src/store/workflow");
    const { Editor } = await import("../src/components/Editor");
    const { tpOptional } = await import("../src/i18n");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    await useWorkflow.getState().startRun();
    const { container } = render(<Editor />);
    act(() => emit({ type: "node_status", node_id: id, status: "running" }));
    act(() =>
      emit({
        type: "node_state",
        node_id: id,
        state: "waiting_memory",
        reason: "gpu_alloc_transient",
        retry_in_s: 120,
      }),
    );
    expect(useWorkflow.getState().run?.nodes[id]?.waiting).toMatchObject({
      reason: "gpu_alloc_transient",
      retry_in_s: 120,
    });
    const badge = container.querySelector('[data-waiting="waiting_memory"]');
    expect(badge?.textContent).toContain("Chờ bộ nhớ trống");
    const copy = tpOptional(["node_state", "waiting_reason", "gpu_alloc_transient"], {
      seconds: 120,
    });
    expect(copy).toBeTruthy(); // design 2dd549b
    expect(badge?.getAttribute("title")).toContain(copy as string);
    expect(copy).toContain("120");
    expect(badge?.textContent ?? "").not.toMatch(/gpu_alloc_transient|\{\w+\}/);
    expect(badge?.getAttribute("title") ?? "").not.toMatch(/gpu_alloc_transient|\{\w+\}/);
    // backend's locked order: running → waiting(reason, 120 s) → running (new attempt) → completed; never a failure shown
    const selectedBefore = useWorkflow.getState().selectedId;
    act(() => emit({ type: "node_status", node_id: id, status: "running" }));
    expect(container.querySelector("[data-waiting]")).toBeNull();
    act(() => emit({ type: "node_status", node_id: id, status: "completed" }));
    expect(useWorkflow.getState().run?.nodes[id]?.status).toBe("completed");
    expect(useWorkflow.getState().selectedId).toBe(selectedBefore); // no failure moved the selection
    expect(useWorkflow.getState().run?.nodes[id]?.error).toBeUndefined();
    expect(container.querySelector('[data-testid="run-error"]')).toBeNull();
  });
});
