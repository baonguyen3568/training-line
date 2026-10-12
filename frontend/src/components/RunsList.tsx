import { useEffect, useState } from "react";
import { api } from "../api";
import type { RunInfo } from "../api/types";
import { t } from "../i18n";

export function RunsList({ onOpen }: { onOpen: (id: string) => void }) {
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    api.runs().then(setRuns, () => setError(true));
  }, []);
  if (error) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!runs) return <div className="skeleton-grid" aria-busy="true" />;
  return (
    <main className="gallery">
      <h1>{t("run_monitor.runs_title")}</h1>
      {runs.length === 0 && <p className="muted">{t("run_monitor.no_runs")}</p>}
      <table className="runs-table">
        <thead>
          <tr>
            <th>{t("run_monitor.col_name")}</th>
            <th>{t("run_monitor.col_status")}</th>
            <th>{t("run_monitor.col_started")}</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className={`run-${r.status}`}>
              <td>
                <button type="button" className="link" onClick={() => onOpen(r.run_id)}>
                  {r.name ?? r.run_id}
                </button>
              </td>
              <td>{t(`run_monitor.status.${r.status}`)}</td>
              <td>{new Date(r.started_at * 1000).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
