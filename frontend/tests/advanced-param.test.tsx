// P016 §3 `advanced` (object, x-ui.widget "advanced"): no widget yet, so the inspector shows nothing for it rather
// than "[object Object]" in a text box; the value stays untouched.
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("advanced engine options param", () => {
  it("is not rendered as a text field", async () => {
    const { useUi } = await import("../src/store/ui");
    useUi.getState().setMode("advanced");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const specs = new Map(useWorkflow.getState().specs);
    const spec = structuredClone(specs.get("train.sft_vlm"));
    if (!spec) throw new Error("no spec");
    spec.params_schema.properties.advanced = {
      type: "object",
      default: {},
      "x-ui": { widget: "advanced", advanced: true },
    } as never;
    specs.set(spec.type_id, spec);
    useWorkflow.setState({ specs });
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    useWorkflow.getState().setParam(id, "advanced", { "optim.lr": 1e-5 });
    useWorkflow.getState().select(id);
    const { Inspector } = await import("../src/components/Inspector");
    const { container } = render(<Inspector />);
    expect(container.querySelector("#param-advanced")).toBeNull();
    expect(container.querySelector('[popovertarget="help-advanced"]')).toBeNull();
    expect(container.textContent).not.toContain("[object Object]");
    expect(useWorkflow.getState().workflow?.nodes[0]?.params.advanced).toEqual({
      "optim.lr": 1e-5,
    });
  });
});
