// Backend `estimate.duration`: measured train time shows even when RAM/disk are unknown; the "?" says where it
// comes from and which heavy steps are not counted yet.
import { render, screen } from "@testing-library/react";
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
