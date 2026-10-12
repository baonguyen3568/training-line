// Guards design review M1: axis tick labels must use formatNumber (lr 1e-4 must not render as "0").
import { render } from "@testing-library/react";
import { expect, it, vi } from "vitest";

const captured: { opts?: { axes: { values: (u: unknown, t: number[]) => string[] }[] } } = {};
vi.mock("uplot", () => ({
  default: class {
    constructor(opts: typeof captured.opts) {
      captured.opts = opts;
    }
    destroy() {}
    setSize() {}
    setLegend() {}
  },
}));

it("formats axis ticks with formatNumber", async () => {
  const { LineChart } = await import("../src/components/LineChart");
  render(
    <LineChart
      title="lr"
      x={[0, 1]}
      series={[{ label: "", values: [0.0001, 0.0001], colorVar: "--x" }]}
      xLabel="step"
    />,
  );
  const [xAxis, yAxis] = captured.opts?.axes ?? [];
  expect(yAxis?.values(null, [0, 0.0001, 0.0002])).toEqual(["0", "1.0e-4", "2.0e-4"]);
  expect(xAxis?.values(null, [0, 10])).toEqual(["0", "10"]);
});
