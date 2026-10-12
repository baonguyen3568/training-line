import { t, tpOptional } from "../i18n";
import { identifierView } from "../lib/safeText";

type Warnings = string[] | null | undefined;
// Logs are scanned only after the job ends, so null on a live run means "later", not "never scanned".
const LIVE = new Set(["queued", "running"]);

/** Copy for one warning code; unknown codes fall back to a generic message naming the code. */
function warningCopy(code: string): { title: string; what: string } {
  const shown = identifierView(code).text;
  return {
    title: tpOptional(["engine_warning", code, "title"]) ?? t("engine_warning.generic.title"),
    what:
      tpOptional(["engine_warning", code, "what"]) ??
      t("engine_warning.generic.what", { code: shown }),
  };
}

/**
 * Compact cell for the runs list. null/absent = never scanned (older run) and must NOT read as clean;
 * [] = scanned and clean.
 */
export function EngineWarningsCell({ warnings, status }: { warnings: Warnings; status: string }) {
  if (warnings == null && LIVE.has(status))
    return <span className="muted">— {t("engine_warning.pending")}</span>;
  if (warnings == null) return <span className="muted">? {t("engine_warning.not_scanned")}</span>;
  if (warnings.length === 0)
    return <span className="state ready">✓ {t("engine_warning.clean")}</span>;
  return (
    <span className="state warn" title={warnings.map((c) => warningCopy(c).title).join("\n")}>
      ⚠ {t("engine_warning.count", { warning_count: warnings.length })}
    </span>
  );
}

/** Run detail: one card per warning (icon + text, never color alone). */
export function EngineWarningsPanel({ warnings, status }: { warnings: Warnings; status: string }) {
  if (warnings == null && LIVE.has(status)) return null;
  if (warnings == null)
    return <p className="muted small engine-warnings">? {t("engine_warning.not_scanned")}</p>;
  if (warnings.length === 0) return null;
  return (
    <section className="engine-warnings" aria-label={t("engine_warning.heading")}>
      <h2>⚠ {t("engine_warning.heading")}</h2>
      <ul>
        {warnings.map((code) => {
          const copy = warningCopy(code);
          return (
            <li key={code} className="error-card warn" data-code={identifierView(code).text}>
              <strong>{copy.title}</strong>
              <p>{copy.what}</p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
