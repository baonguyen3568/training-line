// D4 reference pipelines gallery (P4-1 data): only available app templates open; the rest link to the source.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("PipelineGallery", () => {
  it("one card per method: opens only available app templates, never invents numbers, filters by kind", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    const { t } = await import("../src/i18n");
    render(<PipelineGallery />);
    // counts follow research's data file, never a fixed number
    const data = (await import("../src/generated/reference-pipelines.json")).default as unknown as {
      pipelines: {
        id: string;
        method: string;
        kind: string;
        status: string;
        template_id: string | null;
        measured_with?: { model: string } | null;
        measured?: { train_minutes?: number | null; peak_ram_gb?: number | null } | null;
      }[];
    };
    const cards = screen.getAllByRole("listitem").filter((li) => li.classList.contains("rp-card"));
    expect(cards.length).toBe(data.pipelines.length);
    const openable = cards.filter((c) => within(c).queryByRole("button", { name: "Mở" }));
    expect(openable.length).toBe(
      data.pipelines.filter((p) => p.status === "available" && p.template_id).length,
    );
    for (const p of data.pipelines) {
      const card = document.getElementById(`rp-${p.id}`) as HTMLElement;
      // the method is the title; its architecture is one labelled image built from the data
      expect(card.querySelector("h2")?.textContent).toBe(t(`method.${p.method}.title`));
      expect(card.querySelector("svg.arch-diagram[role=img]")?.getAttribute("aria-label")).toMatch(
        /\S/,
      );
      // numbers only where research measured a real run, never invented elsewhere
      const text = card.textContent ?? "";
      if (p.measured?.train_minutes) expect(text).toMatch(/Đo trên DGX Spark · ~\d/);
      else if (p.measured?.peak_ram_gb) expect(text).toContain("Đo trên DGX Spark · RAM đỉnh");
      else expect(text).not.toContain("Đo trên DGX Spark");
      if (p.measured_with)
        expect(text).toContain(t("pipelines.measured_with", { model: p.measured_with.model }));
    }
    const invoice = document.getElementById("rp-invoice-to-json")?.textContent ?? "";
    expect(invoice).toContain("~1 giờ 56 phút · RAM đỉnh 23.7 GB · F1 theo trường 70.97");
    // research compared it with the LoRA run on the same benchmark: the "?" says so, by the other's method
    expect(invoice).toContain(
      t("pipelines.score_compare", { other: t("method.lora.title"), other_value: "86.42" }),
    );
    expect(invoice).toContain("một lần chạy");
    expect(invoice).toContain(t("pipelines.swap_hint")); // only in the popover, never a line of its own
    const external = within(cards[cards.length - 1] as HTMLElement).getByRole("link", {
      name: /Xem nguồn/,
    });
    expect(external.getAttribute("rel")).toBe("noopener noreferrer");
    fireEvent.click(screen.getByRole("tab", { name: /Ứng viên/ }));
    expect(document.querySelectorAll(".rp-card").length).toBe(
      data.pipelines.filter((p) => p.kind === "candidate").length,
    );
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

  it("under 'Đo trên DGX Spark' the card never shows the server's time, even one it measured here", async () => {
    // measured on the user's machine (maybe not a DGX Spark): that belongs to the Inspector, not this card
    const { api } = await import("../src/api");
    const real = await api.templates();
    const base = real[0] as (typeof real)[number];
    vi.spyOn(api, "templates").mockResolvedValue([
      ...real.filter((x) => x.id !== "quantize_bench"),
      {
        ...base,
        id: "quantize_bench",
        estimate: { ...base.estimate, minutes: { lo: 25, hi: 25 }, source: "measured" },
      },
    ]);
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    render(<PipelineGallery />);
    await new Promise((r) => setTimeout(r, 50));
    const card = document.getElementById("rp-quantize-fp8-benchmark")?.textContent ?? "";
    expect(card).toContain("Đo trên DGX Spark · RAM đỉnh 17.8 GB");
    expect(card).not.toContain("25 phút");
  });

  it("the compared pipeline's name brings its card into view, even when the search hides it", async () => {
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    render(<PipelineGallery />);
    // the LoRA card stays, the Full SFT card it compares with is filtered out
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "LoRA" } });
    expect(document.getElementById("rp-invoice-to-json")).toBeNull();
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    const link = screen
      .getAllByRole("button", { hidden: true }) // inside a closed "?" popover
      .find(
        (b) =>
          b.classList.contains("link") &&
          b.closest("p")?.textContent?.includes("Trên cùng bộ kiểm tra"),
      );
    if (!link) throw new Error("no compare link");
    fireEvent.click(link);
    await waitFor(() => expect(document.getElementById("rp-invoice-to-json")).toBeTruthy());
    await waitFor(() => expect(scroll).toHaveBeenCalled());
  });
});
