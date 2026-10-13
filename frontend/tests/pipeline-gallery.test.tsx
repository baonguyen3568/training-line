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
    // counts follow research's data file (pipelines are added over time), never a fixed number
    const data = (await import("../src/generated/reference-pipelines.json")).default as {
      pipelines: {
        id: string;
        kind: string;
        status: string;
        template_id: string | null;
        title: { vi: string };
        measured?: { train_minutes?: number } | null;
      }[];
    };
    const cards = screen.getAllByRole("listitem").filter((li) => li.classList.contains("rp-card"));
    expect(cards.length).toBe(data.pipelines.length);
    const openable = cards.filter((c) => within(c).queryByRole("button", { name: "Mở" }));
    // only available app templates open (e.g. quantize_bench stays "planned")
    expect(openable.length).toBe(
      data.pipelines.filter((p) => p.status === "available" && p.template_id).length,
    );
    // numbers only where research measured a real run (data file), never invented elsewhere
    for (const p of data.pipelines) {
      const card = cards.find((c) => c.querySelector("h2")?.textContent === p.title.vi);
      if (p.measured?.train_minutes) expect(card?.textContent).toMatch(/Thời gian~\d/);
      else expect(card?.textContent).toContain("chưa có số đo");
    }
    const parseTitle = data.pipelines.find((p) => p.id === "invoice-to-json")?.title.vi;
    const invoice = cards.find((c) => c.querySelector("h2")?.textContent === parseTitle);
    expect(invoice?.textContent).toContain(
      "~1 giờ 56 phút · RAM đỉnh 23.7 GB · F1 theo trường 70.97",
    );
    // research compared it with the Qwen LoRA on the same benchmark: the "?" says so, by the other's title
    expect(invoice?.textContent).toMatch(/Trên cùng bộ kiểm tra, pipeline ".+" đạt 86\.42\./);
    expect(invoice?.textContent).toContain("một lần chạy");
    const external = within(cards[cards.length - 1] as HTMLElement).getByRole("link", {
      name: /Xem nguồn/,
    });
    expect(external.getAttribute("rel")).toBe("noopener noreferrer");
    fireEvent.click(screen.getByRole("tab", { name: /Ứng viên/ }));
    expect(document.querySelectorAll(".rp-card").length).toBe(
      data.pipelines.filter((p) => p.kind === "candidate").length,
    );
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
