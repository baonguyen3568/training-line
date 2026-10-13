// Backend `estimate.duration`: measured train time shows even when RAM/disk are unknown; the "?" says where it
// comes from and which heavy steps are not counted yet.
import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("measured duration", () => {
  it("pre-flight line and D4 summary show the measured range with its basis", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { CheckBar } = await import("../src/components/CheckBar");
    const { Inspector } = await import("../src/components/Inspector");
    const { ReactFlowProvider } = await import("@xyflow/react");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const train = useWorkflow.getState().addNode("train.lora_vlm", { x: 0, y: 0 }) as string;
    const evalId = useWorkflow.getState().addNode("eval.benchmarks", { x: 300, y: 0 }) as string;
    useUi.setState({ skin: "d4" });
    useWorkflow.setState({
      selectedId: null,
      validation: {
        ok: true,
        errors: [],
        estimate: {
          ram_gb_peak: null,
          disk_gb: null,
          duration_min: { lo: 97, hi: 112 },
          gpu_exclusive: true,
          source: "unknown",
          duration: {
            duration_min: { lo: 97, hi: 112 },
            per_node: {
              [train]: { steps: 300, minutes: { lo: 97, hi: 112 }, based_on: ["r1", "r2"] },
            },
            unmeasured: [evalId],
          },
        },
      },
    });
    render(
      <ReactFlowProvider>
        <CheckBar />
        <Inspector />
      </ReactFlowProvider>,
    );
    const bar = screen.getByTestId("check-bar").textContent ?? "";
    expect(bar).toContain("Dự kiến: ~1 giờ 37 phút – 1 giờ 52 phút · Chưa có số đo: RAM, ổ đĩa");
    expect(bar.match(/chưa có số đo/gi)).toHaveLength(1);
    const pop = document.getElementById("est-duration-help");
    expect(pop?.textContent).toContain("Đã đo trên máy này, từ 2 lần chạy trước");
    expect(pop?.textContent).toMatch(/Chưa tính thời gian của: .+\./);
    expect(pop?.textContent).not.toContain(evalId);
  });
});

describe("measured duration of one train step", () => {
  it("shows minutes and steps for the selected node, and per-epochs time in the picker", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { useUi } = await import("../src/store/ui");
    const { Inspector } = await import("../src/components/Inspector");
    useUi.getState().setMode("advanced");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const train = useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }) as string;
    useWorkflow.getState().select(train);
    useWorkflow.setState({
      validation: {
        ok: true,
        errors: [],
        estimate: {
          ram_gb_peak: null,
          disk_gb: null,
          duration_min: { lo: 97, hi: 112 },
          gpu_exclusive: true,
          source: "unknown",
          duration: {
            duration_min: { lo: 97, hi: 112 },
            per_node: {
              [train]: {
                steps: 300,
                minutes: { lo: 97, hi: 112 },
                based_on: ["r1"],
                by_epochs: { "1": { lo: 38.5, hi: 39.4 }, "3": { lo: 114, hi: 115 }, "5": null },
              },
            },
            unmeasured: [],
          },
        },
      },
    });
    const { container } = render(<Inspector />);
    expect(screen.getByTestId("node-estimate").textContent).toBe(
      " ~1 giờ 37 phút – 1 giờ 52 phút · 300 bước học",
    );
    const opts = [...container.querySelectorAll<HTMLOptionElement>("#param-epochs option")];
    expect(opts.find((o) => o.value === "1")?.textContent).toBe("1 lượt · ~39 phút");
    expect(opts.find((o) => o.value === "3")?.textContent).toMatch(/^3 lượt · ~1 giờ 55 phút/);
    expect(opts.find((o) => o.value === "5")?.textContent).toBe("5 lượt"); // no measurement: nothing invented
    // Simple mode: time only, no raw step count
    act(() => useUi.getState().setMode("simple"));
    expect(screen.getByTestId("node-estimate").textContent).toBe(" ~1 giờ 37 phút – 1 giờ 52 phút");
  });
});
