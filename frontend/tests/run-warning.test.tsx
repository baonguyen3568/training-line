// Run `warning` events (CACHE-001 on re-run into an old run dir) must reach the node face and the check bar.
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RunEvent } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

describe("run warnings", () => {
  it("keeps CACHE-001 on the node across later status events and shows it in the check bar", async () => {
    const { api } = await import("../src/api");
    const { useWorkflow } = await import("../src/store/workflow");
    const { CheckBar } = await import("../src/components/CheckBar");
    const { ReactFlowProvider } = await import("@xyflow/react");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const s = () => useWorkflow.getState();
    const id = s().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    let emit: (ev: RunEvent) => void = () => {};
    vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "r1" });
    vi.spyOn(api, "watchRun").mockImplementation((_id, onEvent) => {
      emit = onEvent;
      return () => {};
    });
    await s().startRun();
    emit({
      type: "warning",
      node_id: id,
      error: { code: "CACHE-001", params: { node: id, saved_at: 1791795518, est_minutes: null } },
    } as RunEvent);
    emit({ type: "node_status", node_id: id, status: "running" } as RunEvent);
    expect(s().run?.nodes[id]?.status).toBe("running");
    expect(s().run?.nodes[id]?.warnings?.[0]?.code).toBe("CACHE-001");

    render(
      <ReactFlowProvider>
        <CheckBar />
      </ReactFlowProvider>,
    );
    const btn = await waitFor(() => screen.getByTestId("run-warning"));
    expect(btn.textContent).toContain("không đọc được nên app sẽ chạy lại"); // what_no_estimate variant
    expect(btn.textContent).toContain("Chọn model");
    expect(btn.textContent).not.toMatch(/null|undefined|model\.hf_pick-/);
    vi.restoreAllMocks();
  });
});
