// Run monitor on the real S-3 smoke run fixture (AutoModel 26.08 → contract metrics.jsonl).
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// uPlot needs a real canvas; jsdom has none. Data transforms are tested in src/lib/metrics.test.ts.
vi.mock("uplot", () => ({
  default: class {
    destroy() {}
    setSize() {}
    setLegend() {}
  },
}));
vi.stubEnv("VITE_MOCK_API", "1");

describe("RunMonitor", () => {
  it("shows tiles from the real run and one chart per measure", async () => {
    const { RunMonitor } = await import("../src/components/RunMonitor");
    const { container } = render(<RunMonitor runId="20261012-1630-lora-qwen3.5-0.8b-s3a" />);
    await waitFor(() => expect(screen.getByText(/S-3 smoke LoRA Qwen3.5-0.8B/)).toBeTruthy());
    await waitFor(() => expect(container.querySelectorAll("figure.chart")).toHaveLength(5));
    const values = [...container.querySelectorAll(".tile-value")].map((e) => e.textContent);
    expect(values[0]).toBe("20"); // steps done (0-based last step 19)
    expect(values[1]).toBe("1.42"); // last loss 1.4218
    expect(values[3]).toBe("8.1 GB"); // peak job memory (step 0 warm-up), not machine memory
    expect(container.querySelectorAll(".table-view tbody tr")).toHaveLength(20);
    // lr 1e-4 must never render as "0" (design review r3 M1).
    expect(container.querySelector(".table-view tbody tr td:nth-child(3)")?.textContent).toBe(
      "1.0e-4",
    );
  });
});

describe("RunMonitor paging", () => {
  it("drains every page of a completed run (has_more), not just the first", async () => {
    const { api } = await import("../src/api");
    const total = 12_345;
    const run = {
      schema_version: 1,
      run_id: "big",
      name: "big run",
      stage: "train" as const,
      task: "sft",
      status: "completed" as const,
      started_at: 0,
    };
    vi.spyOn(api, "run").mockResolvedValue(run);
    const calls: number[] = [];
    vi.spyOn(api, "runMetrics").mockImplementation(async (_id, q) => {
      const from = Number(q.after ?? 0);
      calls.push(from);
      const n = Math.min(q.limit, total - from);
      const points = Array.from({ length: n }, (_, i) => ({
        schema_version: 1,
        ts: from + i,
        step: from + i,
        loss: 1 / (from + i + 1),
      }));
      return {
        points,
        next_cursor: String(from + n),
        has_more: from + n < total,
        downsampled: false,
      };
    });
    const { RunMonitor } = await import("../src/components/RunMonitor");
    const { container } = render(<RunMonitor runId="big" />);
    await waitFor(() =>
      expect(container.querySelector(".tile-value")?.textContent).toBe(String(total)),
    );
    expect(calls).toEqual([0, 5000, 10000]);
  });
});

describe("RunsList", () => {
  it("hides engineering spike runs in Simple mode and shows them in Advanced", async () => {
    const { RunsList } = await import("../src/components/RunsList");
    const { useUi } = await import("../src/store/ui");
    const { api } = await import("../src/api");
    vi.restoreAllMocks();
    const runsSpy = vi.spyOn(api, "runs");
    useUi.getState().setMode("simple");
    const view = render(<RunsList onOpen={() => {}} />);
    await waitFor(() => expect(runsSpy).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/run_monitor.no_runs|Chưa có/)).toBeTruthy());
    expect(view.container.querySelector(".runs-table")).toBeNull(); // no empty header row
    expect(screen.queryByText(/MedPix, spike only/)).toBeNull();
    useUi.getState().setMode("advanced");
    await waitFor(() => expect(screen.getByText(/MedPix, spike only/)).toBeTruthy());
  });
});
