// Pipeline score numbers (design 94ee3e9): a difference metric is signed with a real minus, its interval reads
// "−1.8 – 0.4"; the quantize card shows the delta, explains it first in its "?", and leaves out the time it lacks.
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.stubEnv("VITE_MOCK_API", "1");

describe("score format", () => {
  it("signs differences with U+2212, keeps F1 at two decimals", async () => {
    const { formatScore, formatCi } = await import("../src/lib/score");
    const d = "accuracy_delta_fp8_minus_bf16";
    expect(formatScore(d, -0.6)).toBe("−0.6");
    expect(formatScore(d, 0.3)).toBe("+0.3");
    expect(formatScore(d, 0.04)).toBe("0.0");
    expect(formatScore(d, -0.04)).toBe("0.0"); // never "−0.0"
    expect(formatCi(d, [-1.8, 0.4])).toBe("−1.8 – 0.4");
    expect(formatScore("field_f1", 70.97)).toBe("70.97");
    expect(formatCi("field_f1", [66.33, 75.38])).toBe("66.33–75.38");
    expect(formatScore(d, -0.6)).not.toContain("-"); // ASCII hyphen-minus
  });

  it("the FP8 card shows the signed delta, its help first in the popover, and no 'no measurement' time", async () => {
    const { useWorkflow } = await import("../src/store/workflow");
    await useWorkflow.getState().loadCatalog();
    const { PipelineGallery } = await import("../src/components/PipelineGallery");
    const { t } = await import("../src/i18n");
    render(<PipelineGallery />);
    const card = document.getElementById("rp-quantize-fp8-benchmark");
    if (!card) throw new Error("no quantize-fp8-benchmark card");
    const text = card.textContent ?? "";
    expect(text).toContain(`${t("pipelines.metric.accuracy_delta_fp8_minus_bf16")} −0.6`);
    expect(text).not.toContain(t("estimate.source.unknown"));
    const pop = document.getElementById("measured-quantize-fp8-benchmark");
    // the help is the popover's first paragraph (under its title), before the compare / single-run lines
    expect(pop?.querySelector("p")?.textContent).toContain("500");
    expect(screen.getAllByRole("listitem").length).toBeGreaterThan(0);
  });
});
