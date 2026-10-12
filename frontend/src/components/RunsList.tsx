import { useEffect, useState } from "react";
import { api } from "../api";
import type { RunInfo } from "../api/types";
import { t } from "../i18n";
import { formatDateTime } from "../lib/format";
import { useUi } from "../store/ui";
import { EngineWarningsCell } from "./EngineWarnings";

export function RunsList({ onOpen }: { onOpen: (id: string) => void }) {
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [error, setError] = useState(false);
  const mode = useUi((s) => s.mode);
  useEffect(() => {
    api.runs().then(setRuns, () => setError(true));
  }, []);
  if (error) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!runs) return <div className="skeleton-grid" aria-busy="true" />;
  // Engineering spikes (e.g. restricted datasets) are hidden from the no-code view (design review r4).
  const shown = mode === "advanced" ? runs : runs.filter((r) => !r.tags?.includes("spike"));
  return (
    <main className="gallery">
      <h1>{t("run_monitor.runs_title")}</h1>
      {shown.length === 0 && <p className="muted">{t("run_monitor.no_runs")}</p>}
      {shown.length > 0 && (
        <table className="runs-table">
          <thead>
            <tr>
              <th>{t("run_monitor.col_name")}</th>
              <th>{t("run_monitor.col_status")}</th>
              <th>{t("run_monitor.col_started")}</th>
              <th>{t("engine_warning.col")}</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.run_id} className={`run-${r.status ?? "running"}`}>
                <td>
                  <button type="button" className="link" onClick={() => onOpen(r.run_id)}>
                    {r.name ?? r.run_id}
                  </button>
                </td>
                <td>{t(`run_monitor.status.${r.status ?? "running"}`)}</td>
                <td>{r.started_at ? formatDateTime(r.started_at * 1000) : "—"}</td>
                <td>
                  <EngineWarningsCell warnings={r.engine_warnings} status={r.status ?? "running"} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
