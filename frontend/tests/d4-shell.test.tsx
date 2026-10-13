// M2-6 D4 shell (?ui=d4): same editor behaviour, icon-first chrome. Design conditions #2-#4 (2026-10-13).
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

async function setup() {
  const { useUi } = await import("../src/store/ui");
  useUi.setState({ skin: "d4", libraryOpen: false, dockOpen: false });
  const { useWorkflow } = await import("../src/store/workflow");
  await useWorkflow.getState().loadCatalog();
  useWorkflow.getState().newWorkflow();
  const { Editor } = await import("../src/components/Editor");
  const view = render(<Editor />);
  return { useUi, useWorkflow, ...view };
}

describe("D4 shell", () => {
  it("icon-only controls carry their tooltip text as the accessible name", async () => {
    await setup();
    const undo = screen.getByRole("button", { name: "Hoàn tác" });
    expect(undo.getAttribute("data-tip")).toBe("Hoàn tác · Ctrl+Z");
    const mode = screen.getByRole("button", { name: "Bật chế độ Nâng cao" });
    expect(mode.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(mode);
    expect(
      screen.getByRole("button", { name: "Về chế độ Đơn giản" }).getAttribute("aria-pressed"),
    ).toBe("true");
    expect(screen.getByRole("button", { name: /Chạy/ })).toBeTruthy(); // the only labelled control
  });

  it("library flyout: '/' opens it with focus in search, Esc closes and returns focus, adding closes it", async () => {
    const { useUi } = await setup();
    fireEvent.keyDown(window, { key: "/" });
    await waitFor(() => expect(useUi.getState().libraryOpen).toBe(true));
    await waitFor(() => expect((document.activeElement as HTMLInputElement).type).toBe("search"));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(useUi.getState().libraryOpen).toBe(false);
    expect(document.activeElement?.id).toBe("strip-library");
    fireEvent.click(screen.getByRole("button", { name: "Thư viện bước" }));
    fireEvent.doubleClick(screen.getByTestId("library-item-model.hf_pick"));
    await waitFor(() => expect(useUi.getState().libraryOpen).toBe(false));
  });

  it("dock starts collapsed and opens by itself when pre-flight reports errors", async () => {
    const { useUi, useWorkflow } = await setup();
    expect(useUi.getState().dockOpen).toBe(false);
    useWorkflow.getState().addNode("train.sft_vlm", { x: 0, y: 0 }); // missing inputs → CFG-003 errors
    await waitFor(() => expect(useUi.getState().dockOpen).toBe(true), { timeout: 3000 });
  });

  it("compact lanes: empty lanes are 56px and a library drag opens the target lane", async () => {
    const { container } = await setup();
    const widths = () =>
      [...container.querySelectorAll<HTMLElement>(".lane")].map((l) => l.style.width);
    expect(widths()).toEqual(Array(7).fill("56px")); // blank workflow: every lane is empty
    window.dispatchEvent(new CustomEvent("vlm:library-drag", { detail: "train.sft_vlm" }));
    await waitFor(() => expect(widths()[2]).toBe("260px")); // ③ Huấn luyện opens under the drag
    window.dispatchEvent(new CustomEvent("vlm:library-drag", { detail: null }));
    await waitFor(() => expect(widths()[2]).toBe("56px"));
  });
});

describe("D4 status bar", () => {
  it("shows the live run line and a progress bar while a run is going", async () => {
    const { useUi } = await import("../src/store/ui");
    useUi.setState({ skin: "d4" });
    const { useWorkflow } = await import("../src/store/workflow");
    const { setFakeStepMs } = await import("../src/mocks/api");
    setFakeStepMs(200);
    await useWorkflow.getState().loadCatalog();
    useWorkflow.getState().newWorkflow();
    useWorkflow.getState().addNode("model.hf_pick", { x: 0, y: 0 });
    const { StatusBar } = await import("../src/components/ShellD4");
    render(<StatusBar />);
    await useWorkflow.getState().startRun();
    await waitFor(() => expect(screen.getByRole("progressbar")).toBeTruthy(), { timeout: 3000 });
    await waitFor(() => expect(screen.queryByRole("progressbar")).toBeNull(), { timeout: 5000 });
  });
});
