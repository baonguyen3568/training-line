import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { formatNumber } from "../lib/metrics";
import "uplot/dist/uPlot.min.css";

export interface ChartSeries {
  label: string;
  values: (number | null)[];
  /** CSS variable name of a design token, e.g. "--color-series-1". Text never uses series colors. */
  colorVar: string;
  faded?: boolean;
}

interface Props {
  title: string;
  x: number[];
  series: ChartSeries[];
  xLabel: string;
  yLog?: boolean;
  syncKey?: string;
  /** x positions to mark with ✕ (non-finite values). */
  markers?: number[];
  height?: number;
  /** Magnitudes such as memory start at 0 so changes are not exaggerated. */
  yFromZero?: boolean;
}

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * One y-axis per chart (never dual-axis), 2px lines, crosshair + live legend as hover layer,
 * cursor synced across the run's charts. Colors come from design tokens and are re-read on theme change.
 */
export function LineChart({
  title,
  x,
  series,
  xLabel,
  yLog,
  syncKey,
  markers = [],
  height = 180,
  yFromZero,
}: Props) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el || x.length === 0) return;
    let plot: uPlot | null = null;
    const build = () => {
      plot?.destroy();
      const text = css("--color-text-secondary");
      const grid = css("--color-grid");
      const axis = {
        stroke: text,
        grid: { stroke: grid, width: 1 },
        ticks: { stroke: grid, width: 1 },
      };
      const markerPlugin: uPlot.Plugin = {
        hooks: {
          draw: [
            (u) => {
              if (!markers.length) return;
              const ctx = u.ctx;
              ctx.save();
              ctx.fillStyle = css("--color-status-ink-critical");
              ctx.font = `${12 * devicePixelRatio}px sans-serif`;
              for (const mx of markers) {
                const px = u.valToPos(mx, "x", true);
                ctx.fillText("✕", px - 4 * devicePixelRatio, u.bbox.top + 12 * devicePixelRatio);
              }
              ctx.restore();
            },
          ],
        },
      };
      plot = new uPlot(
        {
          width: el.clientWidth || 400,
          height,
          scales: {
            x: { time: false },
            y: yLog
              ? { distr: 3 }
              : yFromZero
                ? { range: (_u, _min, max) => [0, max > 0 ? max * 1.1 : 1] }
                : {},
          },
          // Tick labels go through formatNumber so lr 1e-4 never renders as "0" (design review r3/r4 M1).
          axes: [
            { ...axis, label: xLabel, values: (_u, ticks) => ticks.map((v) => formatNumber(v)) },
            { ...axis, size: 64, values: (_u, ticks) => ticks.map((v) => formatNumber(v)) },
          ],
          cursor: syncKey ? { sync: { key: syncKey } } : {},
          legend: { show: series.length > 1, live: true },
          plugins: [markerPlugin],
          // Show the latest values in the legend before any hover (no "--"), once the plot is ready.
          hooks: { ready: [(u) => u.setLegend({ idx: x.length - 1 })] },
          series: [
            { label: xLabel },
            ...series.map((s) => ({
              label: s.label,
              stroke: css(s.colorVar),
              width: 2,
              alpha: s.faded ? 0.35 : 1,
              points: { show: x.length <= 40, size: 8 }, // markers off for long runs
              spanGaps: false,
              value: (_u: uPlot, v: number | null) => formatNumber(v),
            })),
          ],
        },
        [x, ...series.map((s) => s.values)] as uPlot.AlignedData,
        el,
      );
    };
    build();
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    media?.addEventListener("change", build);
    const ro = new ResizeObserver(() => plot?.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      media?.removeEventListener("change", build);
      ro.disconnect();
      plot?.destroy();
    };
  }, [x, series, xLabel, yLog, syncKey, markers, height, yFromZero]);

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <div ref={host} />
    </figure>
  );
}
