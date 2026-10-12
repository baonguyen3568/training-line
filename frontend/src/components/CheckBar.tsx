import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
import { t } from "../i18n";
import { formatEstimate } from "../lib/format";
import { nodeTitle } from "../lib/nodeText";
import { useWorkflow } from "../store/workflow";

/** Always-visible pre-flight bar (P010 §6): Run is disabled only by blocking errors; warnings never open dialogs. */
export function CheckBar() {
  const { workflow, validation, validating, specs } = useWorkflow();
  const { select, startRun, cancelRun } = useWorkflow.getState();
  const run = useWorkflow((s) => s.run);
  const { setCenter, getNode } = useReactFlow();
  const [open, setOpen] = useState(false);
  if (!workflow) return null;

  const errors = validation?.errors.filter((e) => e.severity === "error") ?? [];
  const warnings = validation?.errors.filter((e) => e.severity === "warn") ?? [];
  const badNodes = new Set(validation?.errors.map((e) => e.node_id).filter(Boolean));
  const total = workflow.nodes.length;
  const est = validation?.estimate;
  const running = !!run && !run.done;
  const states = run ? Object.entries(run.nodes) : [];
  const doneCount = states.filter(([, s]) =>
    ["completed", "cached", "skipped"].includes(s.status),
  ).length;
  const failedCount = states.filter(([, s]) => s.status === "failed").length;
  const current = states.find(([, s]) => s.status === "running");
  const currentNode = current && workflow.nodes.find((n) => n.id === current[0]);
  const progress = currentNode
    ? t("run.progress", {
        done_count: doneCount,
        total_count: states.length,
        node: nodeTitle(specs.get(currentNode.type), currentNode.type),
      })
    : null;
  const finishedText =
    run?.status === "connection_lost" || run?.status === "run_not_found"
      ? t("run.connection_lost")
      : failedCount > 0
        ? t("run.finished_with_errors", { error_count: failedCount })
        : t("node_state.done");

  const showMe = (nodeId: string | undefined) => {
    if (!nodeId) return;
    select(nodeId);
    const n = getNode(nodeId);
    if (n) setCenter(n.position.x + 120, n.position.y + 60, { zoom: 1.1, duration: 300 });
  };

  return (
    <section
      className="checkbar"
      aria-live="polite"
      aria-label={t("preflight.title")}
      data-testid="check-bar"
    >
      <div className="checkbar-row">
        <strong>{t("preflight.title")}:</strong>{" "}
        {validating && !validation ? (
          "…"
        ) : errors.length + warnings.length === 0 ? (
          <span className="ok">✓ {t("preflight.summary_ok")}</span>
        ) : (
          <span>
            {t("preflight.summary", {
              ok_count: total - badNodes.size,
              total_count: total,
              error_count: errors.length,
              warning_count: warnings.length,
            })}
          </span>
        )}
        {est && (
          <span className="muted">
            {" "}
            ·{" "}
            {est.source === "unknown" ||
            (est.duration_min == null && est.ram_gb_peak == null && est.disk_gb == null)
              ? t("estimate.source.unknown_all")
              : t("preflight.estimate_line", {
                  time: formatEstimate(est.duration_min, "minutes", est.source),
                  ram: formatEstimate(est.ram_gb_peak, "ram_gb", est.source),
                  disk: formatEstimate(est.disk_gb, "disk_gb", est.source),
                })}
          </span>
        )}
        {(errors.length > 0 || warnings.length > 0) && (
          <button
            type="button"
            className="link"
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
          >
            {t("preflight.show_details")}
          </button>
        )}
        <span className="spacer" />
        {running && progress && <span data-testid="run-progress">{progress}</span>}
        {run?.done && <span>{finishedText}</span>}
        {running ? (
          <button type="button" className="primary" onClick={() => void cancelRun()}>
            ■ {t("run.stop")}
          </button>
        ) : (
          <button
            type="button"
            className="primary"
            disabled={!validation || errors.length > 0 || workflow.nodes.length === 0}
            title={errors.length > 0 ? t("preflight.run_blocked") : undefined}
            onClick={() => void startRun()}
          >
            ▶ {t("app.run")}
          </button>
        )}
      </div>
      {open && (
        <ul className="issues">
          {[...errors, ...warnings].map((e) => (
            <li
              key={`${e.code}-${e.node_id}-${e.port}-${e.param}-${e.what}`}
              className={`issue ${e.severity}`}
              data-code={e.code}
            >
              <span aria-hidden="true">{e.severity === "error" ? "✕" : "⚠"}</span>
              {/* Server-rendered copy is plain text; React escapes it. */}
              <span>
                <strong>{e.title}</strong> — {e.what}
              </span>
              {e.node_id && (
                <button type="button" className="link" onClick={() => showMe(e.node_id)}>
                  {t("preflight.show_me")}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
