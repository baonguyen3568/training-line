// Vertical slice: gallery → open template → canvas → validate overlay → check bar (S-5).
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("vertical slice", () => {
  it("opens a template from the gallery and shows a clean pre-flight", async () => {
    const { App } = await import("../src/App");
    render(<App />);
    const buttons = await screen.findAllByRole("button", { name: "Dùng mẫu này" });
    fireEvent.click(buttons[0] as HTMLElement);
    await waitFor(() => expect(screen.getByText(/Mọi thứ đã sẵn sàng/)).toBeTruthy());
    expect(screen.getByRole("button", { name: /Chạy/ })).toHaveProperty("disabled", false);
  });

  it("blocks Run when a required input is disconnected via the 'Nhận từ' picker", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    const { Editor } = await import("../src/components/Editor");
    await useWorkflow.getState().loadCatalog();
    await useWorkflow.getState().openTemplate("lora_finetune");
    useWorkflow.getState().select("n4");
    render(<Editor />);
    const pickers = screen.getAllByLabelText(/Nhận từ/);
    fireEvent.change(pickers[0] as HTMLElement, { target: { value: "" } });
    await waitFor(
      () => expect(screen.getByRole("button", { name: /Chạy/ })).toHaveProperty("disabled", true),
      {
        timeout: 2000,
      },
    );
  });
});
