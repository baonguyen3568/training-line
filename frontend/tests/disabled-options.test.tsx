// x-ui.disabled_options (research/design 2026-10-13): values the worker cannot run are dimmed, not selectable,
// still announced; a value already in an old file stays shown and can be changed away.
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

async function inspectQuantize(params: Record<string, unknown>) {
  const { useUi } = await import("../src/store/ui");
  useUi.getState().setMode("advanced");
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  const specs = new Map(useWorkflow.getState().specs);
  const spec = structuredClone(specs.get("quantize.modelopt_ptq"));
  if (!spec) throw new Error("no quantize spec");
  const props = spec.params_schema.properties;
  props.kv_cache_quant = {
    ...props.kv_cache_quant,
    "x-ui": {
      ...props.kv_cache_quant?.["x-ui"],
      disabled_options: { fp8: "param_unsupported.worker_v1" },
    },
  } as never;
  props.quantize_vision = {
    ...props.quantize_vision,
    "x-ui": { ...props.quantize_vision?.["x-ui"], disabled_options: { true: null } },
  } as never;
  specs.set(spec.type_id, spec);
  useWorkflow.setState({ specs });
  useWorkflow.getState().newWorkflow();
  const id = useWorkflow.getState().addNode("quantize.modelopt_ptq", { x: 0, y: 0 }) as string;
  for (const [k, v] of Object.entries(params)) useWorkflow.getState().setParam(id, k, v);
  useWorkflow.getState().select(id);
  const { Inspector } = await import("../src/components/Inspector");
  return render(<Inspector />);
}

describe("disabled_options", () => {
  it("dims fp8 and locks quantize_vision off, with the reason readable", async () => {
    const { container } = await inspectQuantize({});
    const fp8 = [...container.querySelectorAll("option")].find((o) => o.value === "fp8");
    expect(fp8?.disabled).toBe(true);
    expect(fp8?.textContent).toContain("chưa hỗ trợ");
    const box = container.querySelector<HTMLInputElement>("#param-quantize_vision");
    expect(box?.disabled).toBe(true);
    expect(box?.getAttribute("aria-describedby")).toBe("param-quantize_vision-unsupported");
  });

  it("keeps an old file's forbidden value shown and changeable (no silent rewrite)", async () => {
    const { container } = await inspectQuantize({ kv_cache_quant: "fp8", quantize_vision: true });
    const select = container.querySelector<HTMLSelectElement>("#param-kv_cache_quant");
    expect(select?.value).toBe("fp8");
    expect([...(select?.options ?? [])].find((o) => o.value === "fp8")?.disabled).toBe(false);
    const box = container.querySelector<HTMLInputElement>("#param-quantize_vision");
    expect(box?.checked).toBe(true);
    expect(box?.disabled).toBe(false);
    expect(screen.getAllByText(/chưa hỗ trợ/).length).toBeGreaterThan(0);
  });
});
