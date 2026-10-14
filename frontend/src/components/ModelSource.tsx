import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { EventError, ModelResolveJob, UserModel } from "../api/types";
import { t } from "../i18n";
import { hubError, isRepoId, pollHubJob, recentRepos, rememberRepo } from "../lib/hubJob";
import { licenseName } from "../lib/licenses";
import { runErrorText } from "../lib/runWarning";
import { useWorkflow } from "../store/workflow";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";

export type ModelSourceTab = "catalog" | "hub";

/** The three params a user-added model sets together (backend H3); the server refuses them when they disagree. */
export const USER_MODEL_PARAMS = ["user_model", "user_repo", "user_revision"];

/** Pick a user-added model for the step (id, repo, pinned 40-hex revision), or `null` back to the catalog choice. */
export function setUserModel(
  nodeId: string,
  model: Pick<UserModel, "id" | "repo" | "revision"> | null,
) {
  // one edit, one undo step: the three never disagree in between (validate, autosave, Ctrl+Z)
  useWorkflow.getState().setParams(nodeId, {
    user_model: model?.id ?? "",
    user_repo: model?.repo ?? "",
    user_revision: model?.revision ?? "",
  });
}

/** The segmented "Đã kiểm chứng | Từ Hugging Face" (spec H §1, §4), shared by the model and the dataset step. */
export function SourceTabs({
  value,
  onChange,
}: {
  value: ModelSourceTab;
  onChange: (v: ModelSourceTab) => void;
}) {
  return (
    <div className="seg byo-tabs" role="tablist" aria-label={t("byo.source.hub")}>
      {(["catalog", "hub"] as const).map((v) => (
        <button
          key={v}
          type="button"
          role="tab"
          aria-selected={value === v}
          className={value === v ? "on" : ""}
          onClick={() => onChange(v)}
        >
          <Icon name={v === "catalog" ? "shield-check" : "search"} size={14} />
          {t(v === "catalog" ? "byo.source.catalog" : "byo.source.hub")}
        </button>
      ))}
    </div>
  );
}

/**
 * "Từ Hugging Face" for the base model (ADR-021, spec H §1): paste a repo id, check it on Enter or "Kiểm tra" (never
 * per keystroke), see what it is (family · parameters · download size · licence · gated), then "Dùng model này". The
 * model joins the G-5 path: not on this machine → the node's download button and its licence dialog.
 */
export function ModelHubPanel({ nodeId }: { nodeId: string }) {
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("");
  const [job, setJob] = useState<ModelResolveJob | null>(null);
  const [error, setError] = useState<EventError | null>(null);
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  // set on every mount: StrictMode mounts twice in dev, and a ref left false would stop the poll for good
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const recent = recentRepos("model");

  const check = async (value = repo) => {
    const r = value.trim();
    if (!isRepoId(r) || busy) return;
    setRepo(r);
    setBusy(true);
    setError(null);
    setJob(null);
    try {
      const { resolve_id } = await api.addUserModel({
        repo: r,
        ...(revision.trim() ? { revision: revision.trim() } : {}),
      });
      rememberRepo("model", r);
      const end = await pollHubJob(
        () => api.userModelResolve(resolve_id),
        (j) => alive.current && setJob(j),
        () => !alive.current,
      );
      if (end?.status === "failed" && end.error) setError(end.error);
    } catch (e) {
      if (alive.current) setError(hubError(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const model = job?.status === "done" ? job.model : undefined;
  const errText = error ? runErrorText(error, (id) => id) : undefined;

  return (
    <div className="byo-panel">
      <div className="byo-input">
        <input
          type="text"
          className="mono"
          spellCheck={false}
          autoComplete="off"
          aria-label={t("byo.source.hub")}
          placeholder={t("byo.model.placeholder")}
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void check();
          }}
        />
        <button
          type="button"
          className="hairline"
          disabled={!isRepoId(repo) || busy}
          onClick={() => void check()}
        >
          {t("byo.check")}
        </button>
      </div>
      <details className="byo-revision">
        <summary>{t("byo.revision")}</summary>
        <input
          type="text"
          className="mono"
          spellCheck={false}
          aria-label={t("byo.revision")}
          value={revision}
          onChange={(e) => setRevision(e.target.value)}
        />
      </details>
      {recent.length > 0 && !busy && !model && (
        <p className="small muted byo-recent">
          {t("byo.recent")}:{" "}
          {recent.map((r, i) => (
            <span key={r}>
              {i > 0 && ", "}
              <button type="button" className="link mono" onClick={() => void check(r)}>
                {r}
              </button>
            </span>
          ))}
        </p>
      )}
      {busy && (
        <p className="small" role="status">
          <span className="si si-muted spin" aria-hidden="true">
            <Icon name="loader-circle" size={14} />
          </span>{" "}
          {t("byo.checking")}
        </p>
      )}
      {error && (
        <div className="error-card" role="alert" data-code={error.code}>
          <ErrorIcon tone="critical" />
          <strong>{errText?.title}</strong>
          {errText?.what && <p>{errText.what}</p>}
        </div>
      )}
      {model && (
        <div className="byo-result" data-testid="byo-model-result">
          <p>
            <code>{model.repo}</code>{" "}
            <span className="chip unverified">
              <Icon name="badge-question-mark" size={12} /> {t("byo.unverified")}
            </span>
          </p>
          <p className="small">
            {t("byo.model.facts", {
              family: model.family,
              params: `${model.params_b.toFixed(1)}B`,
              size_gb: model.size_gb.toFixed(1),
            })}
          </p>
          <p className="small">
            {t("download.license_lead")}{" "}
            <strong>
              {model.license === "unknown"
                ? t("license.badge.unknown")
                : licenseName(model.license)}
            </strong>
            {" · "}
            <code>{model.revision.slice(0, 12)}</code>
          </p>
          {model.gated && <p className="small">{t("byo.gated")}</p>}
          <p>
            <button type="button" className="primary" onClick={() => setUserModel(nodeId, model)}>
              {t("byo.model.use")}
            </button>
          </p>
        </div>
      )}
    </div>
  );
}
