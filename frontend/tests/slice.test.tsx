// Vertical slice: gallery → open template → canvas → validate overlay → check bar (S-5).
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("vertical slice", () => {
  it("opens a template from the gallery and shows a clean pre-flight (classic)", async () => {
    const { useUi } = await import("../src/store/ui");
    useUi.setState({ skin: "classic" }); // ?ui=classic; the D4 gallery is covered by pipeline-gallery tests
    const { App } = await import("../src/App");
    render(<App />);
    const buttons = await screen.findAllByRole(
      "button",
      { name: "Dùng mẫu này" },
      { timeout: 4000 },
    );
    fireEvent.click(buttons[0] as HTMLElement);
    await waitFor(() => expect(screen.getByText(/Mọi thứ đã sẵn sàng/)).toBeTruthy(), {
      timeout: 4000,
    });
    expect(screen.getByRole("button", { name: /▶ Chạy/ })).toHaveProperty("disabled", false);
  }, 15_000); // full-App integration test: under a parallel full run it can pass 5 s on this shared host

  it("blocks Run when a required input is disconnected via the 'Nhận từ' picker", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    await useWorkflow.getState().openTemplate("image_video_understanding");
    useWorkflow.getState().select("n4");
    render(<Editor />);
    const pickers = await screen.findAllByLabelText(/Nhận từ/);
    fireEvent.change(pickers[0] as HTMLElement, { target: { value: "" } });
    // D4 default: Run lives in the toolbar only (no second one in the dock). Validation is debounced, so wait
    // with the suite's timeout rather than a short one of its own (it went red under load).
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^Chạy/ })).toHaveProperty("disabled", true),
    );
  });

  it("distinguishes train vs val sources in the 'Nhận từ' picker (design review r5 I1)", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    await useWorkflow.getState().openTemplate("invoice_to_json");
    useWorkflow.getState().select("n4");
    render(<Editor />);
    const options = [...document.querySelectorAll("select option")].map((o) => o.textContent ?? "");
    const fromSplit = options.filter((o) => o.startsWith("Chia dữ liệu"));
    expect(new Set(fromSplit).size).toBe(2); // train and val have different labels
    expect(screen.getByText(/chưa có số đo thời gian, RAM, ổ đĩa/)).toBeTruthy();
  });
});
