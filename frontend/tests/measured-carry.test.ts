// React Flow drops a node's handle bounds when it gets a node object without `measured` (tester x100 flake):
// the editor carries the measured size, but only for the same node of the same workflow.
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("measured size carry-over", () => {
  it("is used only for a node of the same type", async () => {
    const { measuredOf } = await import("../src/components/Editor");
    const entry = { type: "train.sft_vlm", width: 240, height: 180 };
    expect(measuredOf(entry, "train.sft_vlm")).toEqual({ width: 240, height: 180 });
    expect(measuredOf(entry, "model.hf_pick")).toBeUndefined(); // id reused by another kind of node
    expect(measuredOf(undefined, "train.sft_vlm")).toBeUndefined();
  });

  it("opening or closing a workflow bumps `opened` (ids n1, n2… repeat across workflows)", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const before = useWorkflow.getState().opened;
    useWorkflow.getState().newWorkflow();
    await useWorkflow.getState().openTemplate("invoice_to_json");
    useWorkflow.getState().closeWorkflow();
    expect(useWorkflow.getState().opened).toBe(before + 3);
  });
});
