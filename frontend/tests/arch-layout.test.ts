// lib/archLayout (spec I §2), shared by the app and the site: every method's diagram fits a 280px card at 1:1 so
// its 11px text is not scaled down, side blocks never overlap, and the label names reference_shared in full.
import { describe, expect, it } from "vitest";
import data from "../src/generated/reference-pipelines.json";
import { t, tp } from "../src/i18n";
import { type ArchShape, archLayout, type TrainArch, type Translate } from "../src/lib/archLayout";

const tr: Translate = (key, params) => (params ? tp(key.split("."), params) : t(key));
const pipelines = (data as unknown as { pipelines: { id: string; train_arch: TrainArch }[] })
  .pipelines;

describe("archLayout", () => {
  it.each(pipelines.map((p) => [p.id, p.train_arch] as const))(
    "%s fits a 280px card without scaling and its side blocks never overlap",
    (_id, arch) => {
      const layout = archLayout(arch, tr);
      expect(layout.width).toBeLessThanOrEqual(286); // 2 stages + labels; the card's content box is ~280
      const sides = layout.groups
        .filter((g) => g.cls.startsWith("arch-side"))
        .map((g) => g.shapes[0] as Extract<ArchShape, { kind: "rect" }>);
      const xs = sides.map((r) => r.x).sort((a, b) => a - b);
      for (let i = 1; i < xs.length; i++)
        expect((xs[i] as number) - (xs[i - 1] as number)).toBeGreaterThanOrEqual(64);
      expect(layout.label).not.toMatch(/arch\./); // every word is copy, never a key
    },
  );

  it("names the shared reference in full in the label, short on the block", () => {
    const layout = archLayout(
      { vision: "frozen", projector: "frozen", lm: "lora", extras: ["reference_shared"] },
      tr,
    );
    expect(layout.label).toContain(t("arch.extra.reference_shared"));
    const text = layout.groups
      .find((g) => g.cls.startsWith("arch-side"))
      ?.shapes.find((s) => s.kind === "text") as Extract<ArchShape, { kind: "text" }>;
    expect(text.lines.join(" ")).toBe(t("arch.extra.reference"));
  });
});
