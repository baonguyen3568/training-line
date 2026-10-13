// x-ui.visible_when (cleancode 038bb8e): data.mix weights only apply to the interleave strategies; a hidden field
// keeps its value and still shows up while it has a validation error.
import { act, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("visible_when", () => {
  it("shows data.mix weights only for interleave strategies, keeping their values", async () => {
    const { useUi } = await import("../src/store/ui");
    useUi.getState().setMode("advanced");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("data.mix", { x: 0, y: 0 }) as string;
    useWorkflow.getState().select(id);
    const { Inspector } = await import("../src/components/Inspector");
    const { container } = render(<Inspector />);
    const weight = () => container.querySelector("#param-weight_1");
    expect(weight()).toBeNull(); // default strategy concat
    act(() => useWorkflow.getState().setParam(id, "weight_1", 3));
    act(() => useWorkflow.getState().setParam(id, "strategy", "interleave_under"));
    expect(weight()).not.toBeNull();
    act(() => useWorkflow.getState().setParam(id, "strategy", "concat"));
    expect(weight()).toBeNull();
    expect(useWorkflow.getState().workflow?.nodes.find((n) => n.id === id)?.params.weight_1).toBe(
      3,
    );
    act(() =>
      useWorkflow.setState({
        validation: {
          ok: false,
          errors: [
            {
              node_id: id,
              param: "weight_1",
              code: "CFG-002",
              severity: "error",
              params: {},
              title: "x",
              what: "Trọng số không hợp lệ",
              action_ids: [],
            },
          ],
          estimate: {
            ram_gb_peak: null,
            disk_gb: null,
            duration_min: null,
            gpu_exclusive: false,
            source: "unknown",
          },
        },
      }),
    );
    expect(weight()).not.toBeNull(); // an error is never hidden
  });
});
