// "Gửi báo cáo lỗi" (design ba32c2c, backend shape B): preview of the 5 safe fields, absent ones left out; copy then
// open the project's issue page; a failed copy selects the text instead. Nothing is sent by the app.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");
afterEach(() => vi.restoreAllMocks());

async function open(report: Record<string, string>) {
  const { api } = await import("../src/api");
  vi.spyOn(api, "bugReport").mockResolvedValue(report);
  const { BugReportDialog } = await import("../src/components/BugReportDialog");
  const onClose = vi.fn();
  render(<BugReportDialog runId="r1" nodeId="n4" code="MEM-003" onClose={onClose} />);
  return { onClose, pre: await screen.findByText(/Mã lỗi: MEM-003/) };
}

describe("bug report", () => {
  it("previews only the fields the server sent", async () => {
    const { pre } = await open({
      code: "MEM-003",
      app_version: "1.0.0+abc1234",
      ref_code: "a1b2c3d4",
    });
    expect(pre.textContent).toMatch(
      /^Mã lỗi: MEM-003\nPhiên bản app: 1\.0\.0\+abc1234\nMã tham chiếu: a1b2c3d4\n\nKhông dán log/,
    ); // the security reminder is the last line of what is copied
    expect(pre.textContent).not.toMatch(/undefined|GPU/);
    expect(screen.getByText(/Không chép nhật ký/)).toBeTruthy();
  });

  it("copies, then opens the issue page with nothing in the URL", async () => {
    const write = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText: write } });
    const win = vi.spyOn(window, "open").mockReturnValue(null);
    const { onClose } = await open({ code: "MEM-003", ref_code: "a1b2c3d4" });
    fireEvent.click(screen.getByRole("button", { name: /Chép và mở GitHub/ }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(write.mock.calls[0]?.[0]).toMatch(
      /^Mã lỗi: MEM-003\nMã tham chiếu: a1b2c3d4\n\nKhông dán log/,
    );
    expect(win).toHaveBeenCalledWith(
      "https://github.com/baonguyen3568/training-line/issues/new",
      "_blank",
      "noopener,noreferrer",
    );
  });

  it("when copying fails, says so and opens nothing", async () => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    const win = vi.spyOn(window, "open").mockReturnValue(null);
    await open({ code: "MEM-003" });
    fireEvent.click(screen.getByRole("button", { name: /Chép và mở GitHub/ }));
    await screen.findByText(/không chép tự động được/);
    expect(win).not.toHaveBeenCalled();
  });

  it("the reminder links SECURITY.md in the preview and keeps the URL as text in the copy", async () => {
    await open({ code: "MEM-003", ref_code: "a1b2c3d4" });
    const link = screen.getByRole("link", { name: /SECURITY\.md/ });
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    expect(document.body.textContent).not.toContain("{security_url}");
  });

  it("an error outside a run reports without a reference code", async () => {
    const { api } = await import("../src/api");
    const spy = vi.spyOn(api, "bugReport");
    const { BugReportDialog } = await import("../src/components/BugReportDialog");
    render(<BugReportDialog runId={null} code="EXP-001" onClose={() => {}} />);
    await screen.findByText(/Mã lỗi: EXP-001/);
    expect(spy).toHaveBeenCalledWith(null, { nodeId: undefined, code: "EXP-001" }, "vi");
    expect(document.body.textContent).not.toContain("Mã tham chiếu:");
  });
});
