// engine_warnings tri-state (run.json): null/absent = not scanned (must not look clean), [] = clean, codes = warnings.
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EngineWarningsCell, EngineWarningsPanel } from "../src/components/EngineWarnings";

describe("engine warnings", () => {
  it("never shows an unscanned run as clean", () => {
    for (const w of [null, undefined]) {
      const { container, unmount } = render(<EngineWarningsCell warnings={w} status="completed" />);
      expect(container.textContent).toContain("Chưa kiểm tra");
      expect(container.textContent).not.toContain("Không có cảnh báo");
      unmount();
    }
    render(<EngineWarningsPanel warnings={null} status="failed" />);
    expect(screen.getByText(/Chưa kiểm tra/)).toBeTruthy();
  });

  it("does not call a live run unscanned: logs are scanned only after the job ends", () => {
    for (const status of ["queued", "running"]) {
      const { container, unmount } = render(<EngineWarningsCell warnings={null} status={status} />);
      expect(container.textContent).toContain("Sẽ kiểm tra khi chạy xong");
      expect(container.textContent).not.toMatch(/Chưa kiểm tra|Không có cảnh báo/);
      unmount();
      const panel = render(<EngineWarningsPanel warnings={null} status={status} />);
      expect(panel.container.innerHTML).toBe("");
      panel.unmount();
    }
  });

  it("shows clean only for an empty list, and nothing extra on the detail page", () => {
    render(<EngineWarningsCell warnings={[]} status="completed" />);
    expect(screen.getByText(/Không có cảnh báo/)).toBeTruthy();
    const { container } = render(<EngineWarningsPanel warnings={[]} status="completed" />);
    expect(container.innerHTML).toBe("");
  });

  it("explains known codes and falls back for unknown ones without rendering markup", () => {
    render(<EngineWarningsPanel warnings={["bf16_adam_params", "<b>x</b>"]} status="completed" />);
    expect(screen.getByText(/không giữ bản fp32/)).toBeTruthy();
    expect(screen.getByText(/<b>x<\/b>/)).toBeTruthy(); // plain text, not HTML
    render(<EngineWarningsCell warnings={["bf16_adam_params"]} status="completed" />);
    expect(screen.getByText(/1 cảnh báo/)).toBeTruthy();
  });
});
