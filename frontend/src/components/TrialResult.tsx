import { useEffect, useState } from "react";
import { api } from "../api";
import { ApiError, uploadUrl } from "../api/client";
import type { TrialResult as Result } from "../api/types";
import { t } from "../i18n";
import { stripInvisible, untrustedText } from "../lib/safeText";

type State =
  | { kind: "loading" }
  | { kind: "ok"; result: Result }
  | { kind: "not_ready" }
  | { kind: "too_large" }
  | { kind: "failed" };

const MAX_TEXT = 20_000; // the server already cuts at 20k and sets truncated

/**
 * Output of a "Chạy thử" node after a run (B-7). Everything in result.json is written by a container: text is
 * shown as plain text (visible escapes for bidi/control chars), decoding/overrides as JSON in <pre>, never HTML.
 */
export function TrialResult({
  runId,
  nodeId,
  imageId,
}: {
  runId: string;
  nodeId: string;
  imageId?: string;
}) {
  const [state, setState] = useState<State>({ kind: "loading" });
  useEffect(() => {
    let live = true;
    setState({ kind: "loading" });
    api.nodeResult(runId, nodeId).then(
      (result) => live && setState({ kind: "ok", result }),
      (e) => {
        if (!live) return;
        const code =
          e instanceof ApiError
            ? (e.body as { error?: { code?: string } } | null)?.error?.code
            : undefined;
        setState({
          kind:
            code === "RESULT_NOT_READY"
              ? "not_ready"
              : code === "RESULT_TOO_LARGE"
                ? "too_large"
                : "failed",
        });
      },
    );
    return () => {
      live = false;
    };
  }, [runId, nodeId]);

  return (
    <section className="trial-result" aria-label={t("trial.result_title")}>
      <h3>{t("trial.result_title")}</h3>
      {state.kind === "loading" && <p className="muted">…</p>}
      {state.kind === "not_ready" && <p className="muted">{t("trial.result_not_ready")}</p>}
      {state.kind === "too_large" && <p className="muted">{t("trial.result_too_large")}</p>}
      {state.kind === "failed" && <p className="muted">{t("app.catalog_load_failed")}</p>}
      {state.kind === "ok" && (
        <>
          {imageId && (
            <img className="trial-image" src={uploadUrl(imageId)} alt={t("trial.image_alt")} />
          )}
          {state.result.truncated && <p className="badge">✂ {t("trial.truncated")}</p>}
          <pre className="trial-text">{untrustedText(state.result.text, MAX_TEXT)}</pre>
          <p className="muted small">
            {t("trial.meta", {
              engine: untrustedText(state.result.engine_used, 20),
              token_count: state.result.n_tokens,
            })}
          </p>
          <details>
            <summary>{t("trial.details")}</summary>
            <pre className="trial-json">
              {stripInvisible(
                JSON.stringify(
                  { decoding: state.result.decoding, overrides: state.result.overrides },
                  null,
                  2,
                ),
              )}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}
