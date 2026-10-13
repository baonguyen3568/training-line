// STD-001 (P4-2) arrives with severity "info": listed in the details, never counted as an error, never blocks Run.
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("severity info", () => {
  it("is listed with the info icon but does not count as an error nor disable Run", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { CheckBar } = await import("../src/components/CheckBar");
    const { ReactFlowProvider } = await import("@xyflow/react");
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    const id = useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 }) as string;
    useWorkflow.setState({
      validation: {
        ok: true,
        errors: [
          {
            node_id: id,
            code: "STD-001",
            severity: "info",
            title: "Theo chuẩn tham khảo",
            what: "Hạng LoRA nằm ngoài khoảng thường dùng.",
            params: {},
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
    });
    // the check bar's own Run button is the classic one (D4 runs from the toolbar)
    const { useUi } = await import("../src/store/ui");
    useUi.setState({ skin: "classic" });
    render(
      <ReactFlowProvider>
        <CheckBar />
      </ReactFlowProvider>,
    );
    expect(screen.getByRole("button", { name: /Chạy/ })).toHaveProperty("disabled", false);
    expect(screen.queryByText(/1 lỗi/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Xem chi tiết/ }));
    const item = document.querySelector("[data-code='STD-001']");
    // the Lucide info icon in muted ink (text glyphs were OS-dependent)
    expect(item?.querySelector(".si-muted svg")).not.toBeNull();
    expect(item?.className).toContain("info");
  });
});
