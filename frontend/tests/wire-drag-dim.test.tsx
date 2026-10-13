// While a wire is dragged, ports it cannot reach fade (design 2026-10-13): same side, own node, other type.
import { act, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("wire drag dims unreachable ports", () => {
  it("keeps compatible inputs of other steps bright and fades the rest", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const model = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    const train = useWorkflow.getState().addNode("train.sft_vlm", { x: 400, y: 0 }) as string;
    const { container } = render(<Editor />);
    const li = (node: string, port: string) =>
      container
        .querySelector(`[data-id="${node}"] .react-flow__handle[data-handleid="${port}"]`)
        ?.closest("li");
    const modelType = useWorkflow.getState().specs.get("model.hf_pick")?.outputs.model?.type;
    if (!modelType) throw new Error("no model output type");
    act(() => useUi.getState().setWireDrag({ nodeId: model, side: "source", type: modelType }));
    expect(li(train, "base_model")?.classList.contains("port-dim")).toBe(false);
    expect(li(train, "train_data")?.classList.contains("port-dim")).toBe(true); // other type
    expect(li(train, "model")?.classList.contains("port-dim")).toBe(true); // same side
    expect(li(model, "model")?.classList.contains("port-dim")).toBe(true); // own node
    act(() => useUi.getState().setWireDrag(null));
    expect(container.querySelectorAll(".port-dim")).toHaveLength(0);
  });

  it("never stays dimmed when the drag ends without onConnectEnd (Esc, window blur)", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const model = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    render(<Editor />);
    const drag = { nodeId: model, side: "source" as const, type: "x" };
    act(() => useUi.getState().setWireDrag(drag));
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(useUi.getState().wireDrag).toBeNull();
    act(() => useUi.getState().setWireDrag(drag));
    act(() => {
      window.dispatchEvent(new Event("blur"));
    });
    expect(useUi.getState().wireDrag).toBeNull();
  });
});
