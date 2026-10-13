import { useReactFlow } from "@xyflow/react";
import { useState } from "react";
import { t } from "../i18n";
import { estimateLine, formatDuration } from "../lib/format";
import { nodeTitle, SEVERITY_GLYPH } from "../lib/nodeText";
import { runWarningText } from "../lib/runWarning";
import { stepNumbers } from "../lib/topo";
import { useWorkflow } from "../store/workflow";
import { Icon } from "./Icon";

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
  // "info" (STD-001 standards, P4-2): shown in the details only, never counted as a problem, never blocks Run.
  const infos = validation?.errors.filter((e) => e.severity === "info") ?? [];
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
  // Step number = the ①②③ shown on the node (design review M1 r2 #1), not a done-counter.
  const progress = currentNode
    ? t("run.progress", {
        current_step: stepNumbers(workflow).get(currentNode.id) ?? doneCount + 1,
        total_count: states.length,
        node: nodeTitle(specs.get(currentNode.type), currentNode.type),
      })
    : null;
  const nodeLabel = (id: string) => {
    const n = workflow.nodes.find((x) => x.id === id);
    return n ? nodeTitle(specs.get(n.type), n.type) : id;
  };
  const runWarnings = states.flatMap(([id, s]) =>
    (s.warnings ?? []).map((w) => ({ id, ...runWarningText(w, nodeLabel) })),
  );
  const lost = run?.status === "connection_lost" || run?.status === "run_not_found";
  const finishedText = lost
    ? t("run.connection_lost")
    : failedCount > 0
      ? t("run.finished_with_errors", { error_count: failedCount })
      : t("run.finished", {
          duration: formatDuration((run?.endedAt ?? Date.now()) - (run?.startedAt ?? Date.now())),
        });

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
        {run ? (
          // While running / after a run, the bar reports the run instead of the pre-flight (r2 #2).
          <>
            {running && (
              <span data-testid="run-progress">{progress ?? t("node_state.queued")}</span>
            )}
            {runWarnings.map((w, i) => (
              <button
                // biome-ignore lint/suspicious/noArrayIndexKey: warnings are append-only for one run
                key={i}
                type="button"
                className="link state warn"
                data-testid="run-warning"
                title={w.what}
                onClick={() => showMe(w.id)}
              >
                <span className="si si-warn" aria-hidden="true">
                  <Icon name="triangle-alert" size={14} />
                </span>{" "}
                {w.what ?? `${w.title}: ${nodeLabel(w.id)}`}
              </button>
            ))}
            {run.done && (
              <>
                <span className={failedCount > 0 || lost ? "run-result bad" : "run-result"}>
                  {failedCount > 0 || lost ? "✕" : "✓"} {finishedText}
                </span>{" "}
                <button
                  type="button"
                  className="link"
                  onClick={() =>
                    window.dispatchEvent(new CustomEvent("vlm:navigate", { detail: "runs" }))
                  }
                >
                  {t("run.view_results")}
                </button>
              </>
            )}
          </>
        ) : (
          <>
            <strong>{t("preflight.title")}:</strong>{" "}
            {total === 0 ? (
              <span className="muted">{t("preflight.empty")}</span>
            ) : validating && !validation ? (
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
            {est && <span className="muted"> · {estimateLine(est)}</span>}
            {(errors.length > 0 || warnings.length > 0 || infos.length > 0) && (
              <button
                type="button"
                className="link"
                aria-expanded={open}
                onClick={() => setOpen((o) => !o)}
              >
                {t("preflight.show_details")}
              </button>
            )}
          </>
        )}
        <span className="spacer" />
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
          {[...errors, ...warnings, ...infos].map((e) => (
            <li
              key={`${e.code}-${e.node_id}-${e.port}-${e.param}-${e.what}`}
              className={`issue ${e.severity}`}
              data-code={e.code}
            >
              <span aria-hidden="true">{SEVERITY_GLYPH[e.severity]}</span>
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
