import { describe, expect, it } from "vitest";
import fixture from "../mocks/fixtures/s3a.metrics.jsonl?raw";
import { ema, formatNumber, lttb, median, nonFiniteSteps, parseJsonl } from "./metrics";

describe("metrics", () => {
  it("parses the real S-3 AutoModel run converted to the contract", () => {
    const { points, skipped } = parseJsonl(fixture);
    expect(points).toHaveLength(20);
    expect(skipped).toBe(0);
    expect(points[0]).toMatchObject({ schema_version: 1, step: 0, split: "train" });
    expect(points.every((p) => typeof p.mem_max_alloc_gb === "number")).toBe(true);
  });

  it("skips a partially written last line", () => {
    const { points, skipped } = parseJsonl('{"ts":1,"step":0,"loss":1}\n{"ts":2,"st');
    expect(points).toHaveLength(1);
    expect(skipped).toBe(1);
  });

  it("smooths with EMA and keeps gaps", () => {
    expect(ema([1, null, 3], 0.5)).toEqual([1, null, 2]);
  });

  it("flags NaN/inf as markers instead of plotting them", () => {
    const pts = [
      { schema_version: 1, ts: 1, step: 0, loss: 1 },
      { schema_version: 1, ts: 2, step: 1, loss: Number.NaN },
    ];
    expect(nonFiniteSteps(pts, "loss")).toEqual([1]);
  });

  it("downsamples with LTTB keeping first and last", () => {
    const xs = Array.from({ length: 10_000 }, (_, i) => i);
    const idx = lttb(
      xs,
      xs.map((x) => Math.sin(x / 50)),
      500,
    );
    expect(idx).toHaveLength(500);
    expect(idx[0]).toBe(0);
    expect(idx.at(-1)).toBe(9999);
  });

  it("formats tiny and huge numbers without rounding to 0", () => {
    expect(formatNumber(0.0001)).toBe("1.0e-4");
    expect(formatNumber(2e-5)).toBe("2.0e-5");
    expect(formatNumber(1.4217815)).toBe("1.42");
    expect(formatNumber(551.01)).toBe("551");
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(Number.NaN)).toBe("—");
  });

  it("computes the median", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(median([])).toBeNull();
  });
});
