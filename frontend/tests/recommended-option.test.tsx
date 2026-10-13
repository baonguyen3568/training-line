// x-ui.recommended (PM/research 2026-10-13): epochs 1|2|3|5 read "N lượt", the recommended one adds "· khuyên dùng".
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("recommended option", () => {
  it("marks only the recommended epochs value", async () => {
    const { useUi } = await import("../src/store/ui");
    useUi.getState().setMode("advanced");
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const specs = new Map(useWorkflow.getState().specs);
    const spec = structuredClone(specs.get("train.lora_vlm"));
    if (!spec) throw new Error("no train spec");
    const props = spec.params_schema.properties;
    props.epochs = {
      ...props.epochs,
      "x-ui": { ...props.epochs?.["x-ui"], recommended: 3 },
    } as never;
    specs.set(spec.type_id, spec);
    useWorkflow.setState({ specs });
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("train.lora_vlm", { x: 0, y: 0 }) as string;
    useWorkflow.getState().select(id);
    const { Inspector } = await import("../src/components/Inspector");
    const { container } = render(<Inspector />);
    const opts = [...container.querySelectorAll<HTMLOptionElement>("#param-epochs option")];
    expect(opts.find((o) => o.value === "3")?.textContent).toBe("3 lượt · khuyên dùng");
    expect(opts.filter((o) => o.textContent?.includes("khuyên dùng"))).toHaveLength(1);
  });
});
