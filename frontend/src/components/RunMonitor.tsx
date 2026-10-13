import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { RunInfo, RunStatus } from "../api/types";
import { t } from "../i18n";
import {
  ema,
  finiteOrNull,
  formatNumber,
  lttb,
  type MetricPoint,
  median,
  nonFiniteSteps,
  SUPPORTED_SCHEMA,
} from "../lib/metrics";
import { EngineWarningsPanel } from "./EngineWarnings";
import { Icon } from "./Icon";
import { type ChartSeries, LineChart } from "./LineChart";

const POLL_MS = 3000;
const MAX_POINTS = 2000; // display budget per chart (client LTTB)
const PAGE_ROWS = 5000; // raw rows per request
// TODO: once the server downsamples (`downsample_to`), keep a bounded window client-side instead of
// accumulating every point of very long runs.
const STALE_S = 120;

const fmt = formatNumber;

const STATUS_ICON: Record<RunStatus, string> = {
  queued: "⧗",
  running: "◐",
  completed: "✓",
  failed: "✕",
  cancelled: "■",
  interrupted: "⏸",
  evicted: "↺",
  discarded: "–",
};

/** Run monitor (P006 annex (c)): is this run healthy? Separate charts per measure, x = step, synced crosshair. */
export function RunMonitor({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunInfo | null>(null);
  const [points, setPoints] = useState<MetricPoint[]>([]);
  const [error, setError] = useState(false);
  const [smooth, setSmooth] = useState(true);
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    let alive = true;
    let cursor: string | undefined;
    let first = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const info = await api.run(runId);
        if (!alive) return;
        setRun(info);
        // Drain every available page regardless of status (finished runs can have far more than one page).
        let more = true;
        while (more && alive) {
          const page = await api.runMetrics(runId, {
            ...(cursor ? { after: cursor } : {}),
            limit: PAGE_ROWS,
            ...(first ? { downsample_to: MAX_POINTS } : {}),
          });
          first = false;
          cursor = page.next_cursor;
          more = page.has_more;
          if (page.points.length) setPoints((prev) => [...prev, ...page.points]);
        }
        setNow(Date.now() / 1000);
        setError(false);
        // Tail only while the run is live or waiting.
        const st = info.status ?? "running";
        if (alive && (st === "running" || st === "queued")) timer = setTimeout(tick, POLL_MS);
      } catch {
        if (alive) {
          setError(true);
          timer = setTimeout(tick, POLL_MS * 3);
        }
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [runId]);

  const train = useMemo(
    () => points.filter((p) => (p.split ?? "train") === "train").sort((a, b) => a.step - b.step),
    [points],
  );
  const unsupported = points.some(
    (p) => !(SUPPORTED_SCHEMA as readonly number[]).includes(p.schema_version),
  );

  const view = useMemo(() => {
    const xsAll = train.map((p) => p.step);
    const lossAll = train.map((p) => finiteOrNull(p.loss));
    const idx = lttb(xsAll, lossAll, MAX_POINTS);
    const pick = <T,>(arr: T[]) => idx.map((i) => arr[i] as T);
    const col = (key: keyof MetricPoint) => pick(train.map((p) => finiteOrNull(p[key])));
    return {
      x: pick(xsAll),
      loss: pick(lossAll),
      lossEma: pick(ema(lossAll, 0.1)),
      lr: col("lr"),
      grad: col("grad_norm"),
      tps: col("tokens_per_s"),
      mem: col("mem_max_alloc_gb"),
      nanSteps: [
        ...new Set([...nonFiniteSteps(train, "loss"), ...nonFiniteSteps(train, "grad_norm")]),
      ],
    };
  }, [train]);

  const lossSeries: ChartSeries[] = useMemo(
    () =>
      smooth
        ? [
            {
              label: t("run_monitor.raw"),
              values: view.loss,
              colorVar: "--color-series-1",
              faded: true,
            },
            {
              label: t("run_monitor.smoothed"),
              values: view.lossEma,
              colorVar: "--color-series-1",
            },
          ]
        : [{ label: t("run_monitor.raw"), values: view.loss, colorVar: "--color-series-1" }],
    [view, smooth],
  );
  const one = (values: (number | null)[]): ChartSeries[] => [
    { label: "", values, colorVar: "--color-series-1" },
  ];

  const last = train.at(-1);
  const stale =
    (run?.status ?? "running") === "running" && last !== undefined && now - last.ts > STALE_S;
  const tpsMedian = median(view.tps.filter((v): v is number => v !== null));
  const memPeak = view.mem.reduce<number | null>(
    (m, v) => (v !== null && (m === null || v > m) ? v : m),
    null,
  );

  if (error && !run) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!run) return <div className="skeleton-grid" aria-busy="true" />;
  const status: RunStatus = run.status ?? "running"; // schema default

  return (
    <main className="run-monitor">
      <header className="run-head">
        <h1>{run.name ?? run.run_id}</h1>
        <span className={`chip status-${status}`}>
          {STATUS_ICON[status]} {t(`run_monitor.status.${status}`)}
        </span>
        {stale && (
          <span className="state warn">
            <span className="si si-warn" aria-hidden="true">
              <Icon name="triangle-alert" size={14} />
            </span>{" "}
            {t("run_monitor.stale", { time: new Date(last.ts * 1000).toLocaleTimeString() })}
          </span>
        )}
        <code className="muted small">{run.run_id}</code>
      </header>
      {unsupported && <p className="banner">{t("run_monitor.schema_unsupported")}</p>}
      <EngineWarningsPanel warnings={run.engine_warnings} status={status} />

      <section className="tiles" aria-label={t("run_monitor.summary")}>
        <div className="tile">
          <span className="tile-label">{t("run_monitor.step")}</span>
          <span className="tile-value">
            {last ? last.step + 1 : "—"}
            {run.max_steps ? ` / ${run.max_steps}` : ""}
          </span>
        </div>
        <div className="tile">
          <span className="tile-label">{t("run_monitor.loss_last")}</span>
          <span className="tile-value">{fmt(finiteOrNull(last?.loss))}</span>
        </div>
        <div className="tile">
          <span className="tile-label">{t("run_monitor.tps_median")}</span>
          <span className="tile-value">{tpsMedian === null ? "—" : Math.round(tpsMedian)}</span>
        </div>
        <div className="tile">
          {/* Job memory (torch max allocated) — NOT whole-machine memory on GB10 unified memory. */}
          <span className="tile-label">{t("run_monitor.mem_peak")}</span>
          <span className="tile-value">{memPeak === null ? "—" : `${memPeak.toFixed(1)} GB`}</span>
        </div>
      </section>

      <label className="check">
        <input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} />
        {t("run_monitor.smoothing")}
      </label>

      {train.length === 0 ? (
        <p className="muted">{t("run_monitor.no_data")}</p>
      ) : (
        <div className="charts">
          <LineChart
            title={t("run_monitor.chart.loss")}
            x={view.x}
            series={lossSeries}
            xLabel={t("run_monitor.step")}
            syncKey={runId}
            markers={view.nanSteps}
          />
          <LineChart
            title={t("run_monitor.chart.lr")}
            x={view.x}
            series={one(view.lr)}
            xLabel={t("run_monitor.step")}
            syncKey={runId}
          />
          <LineChart
            title={t("run_monitor.chart.grad_norm")}
            x={view.x}
            series={one(view.grad)}
            xLabel={t("run_monitor.step")}
            syncKey={runId}
            markers={view.nanSteps}
          />
          <LineChart
            title={t("run_monitor.chart.tps")}
            x={view.x}
            series={one(view.tps)}
            xLabel={t("run_monitor.step")}
            syncKey={runId}
          />
          <LineChart
            title={t("run_monitor.chart.mem")}
            yFromZero
            x={view.x}
            series={one(view.mem)}
            xLabel={t("run_monitor.step")}
            syncKey={runId}
          />
        </div>
      )}

      <details className="table-view">
        <summary>{t("run_monitor.table_view")}</summary>
        <table>
          <thead>
            <tr>
              <th>step</th>
              <th>loss</th>
              <th>lr</th>
              <th>grad_norm</th>
              <th>tokens/s</th>
              <th>mem GB</th>
            </tr>
          </thead>
          <tbody>
            {train.map((p) => (
              <tr key={p.step}>
                <td>{p.step}</td>
                <td>{fmt(finiteOrNull(p.loss), 4)}</td>
                <td>{fmt(finiteOrNull(p.lr), 2)}</td>
                <td>{fmt(finiteOrNull(p.grad_norm), 3)}</td>
                <td>{fmt(finiteOrNull(p.tokens_per_s), 3)}</td>
                <td>{fmt(finiteOrNull(p.mem_max_alloc_gb), 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </main>
  );
}
