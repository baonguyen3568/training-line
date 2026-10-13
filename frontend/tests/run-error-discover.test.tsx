// A step failing while running must be found without hunting (design 1.0): the dock opens and lists it with
// "Chỉ cho tôi", the first failed step is selected when nothing was, a step the user picked is never taken away.
import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RunEvent } from "../src/api/types";

vi.stubEnv("VITE_MOCK_API", "1");

async function start() {
  const { api } = await import("../src/api");
  let emit: (ev: RunEvent) => void = () => {};
  vi.spyOn(api, "startRun").mockResolvedValue({ run_id: "r1" });
  vi.spyOn(api, "watchRun").mockImplementation((_id, onEvent) => {
    emit = onEvent;
    return () => {};
  });
  const { useWorkflow } = await import("../src/store/workflow");
  const { useUi } = await import("../src/store/ui");
  const { Editor } = await import("../src/components/Editor");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const a = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
  const b = useWorkflow.getState().addNode("train.sft_vlm", { x: 300, y: 0 }) as string;
  useWorkflow.getState().select(null);
  await useWorkflow.getState().startRun();
  const view = render(<Editor />);
  const fail = (id: string) =>
    act(() =>
      emit({
        type: "node_status",
        node_id: id,
        status: "failed",
        error: { code: "MEM-003", params: {}, title: "Máy hết bộ nhớ khi đang train", what: "…" },
      }),
    );
  return { useWorkflow, useUi, a, b, fail, ...view };
}

describe("finding a runtime failure", () => {
  it("opens the dock, lists the failed step and selects it when nothing was selected", async () => {
    const { useWorkflow, useUi, b, fail } = await start();
    useUi.setState({ dockOpen: false });
    fail(b);
    expect(useWorkflow.getState().selectedId).toBe(b);
    expect(useUi.getState().dockOpen).toBe(true);
    const row = screen.getByTestId("run-error");
    expect(row.textContent).toContain("Máy hết bộ nhớ khi đang train");
    expect(row.textContent).toContain("Chỉ cho tôi");
  });

  it("never takes the selection away from a step the user picked", async () => {
    const { useWorkflow, a, b, fail } = await start();
    act(() => useWorkflow.getState().select(a));
    fail(b);
    expect(useWorkflow.getState().selectedId).toBe(a);
  });
});
