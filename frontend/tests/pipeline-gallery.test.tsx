// D4 reference pipelines gallery (P4-1 data): only available app templates open; the rest link to the source.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("PipelineGallery", () => {
  it("opens only available app templates, never invents numbers, and filters by kind", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    render(<PipelineGallery />);
    const cards = screen.getAllByRole("listitem").filter((li) => li.classList.contains("rp-card"));
    expect(cards.length).toBe(10);
    const openable = cards.filter((c) => within(c).queryByRole("button", { name: "Mở" }));
    expect(openable.length).toBe(2); // invoice + image/video; quantize_bench is still "planned"
    // numbers only where research measured a real run (data file), never invented elsewhere
    const data = (await import("../src/generated/reference-pipelines.json")).default as {
      pipelines: { title: { vi: string }; measured?: { train_minutes?: number } | null }[];
    };
    for (const p of data.pipelines) {
      const card = cards.find((c) => c.querySelector("h2")?.textContent === p.title.vi);
      if (p.measured?.train_minutes) expect(card?.textContent).toMatch(/Thời gian~\d/);
      else expect(card?.textContent).toContain("chưa có số đo");
    }
    const invoice = cards.find((c) => c.textContent?.includes("Đọc hoá đơn"));
    expect(invoice?.textContent).toContain("~1 giờ 56 phút · RAM đỉnh 23.7 GB");
    const external = within(cards[cards.length - 1] as HTMLElement).getByRole("link", {
      name: /Xem nguồn/,
    });
    expect(external.getAttribute("rel")).toBe("noopener noreferrer");
    fireEvent.click(screen.getByRole("tab", { name: /Ứng viên/ }));
    expect(document.querySelectorAll(".rp-card").length).toBe(3);
    expect(document.body.textContent).toContain("Built on NVIDIA Cosmos");
  });

  it("research's measurement wins over the server's estimate on the card (template defaults)", async () => {
    const { api } = await import("../src/api");
    const real = await api.templates();
    vi.spyOn(api, "templates").mockResolvedValue(
      real.map((x) =>
        x.id === "invoice_to_json"
          ? { ...x, estimate: { ...x.estimate, minutes: { lo: 300, hi: 310 } } }
          : x,
      ),
    );
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    render(<PipelineGallery />);
    await waitFor(() => expect(document.body.textContent).toContain("~1 giờ 56 phút"));
    expect(document.body.textContent).not.toContain("5 giờ");
  });
});
