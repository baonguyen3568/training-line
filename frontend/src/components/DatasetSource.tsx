import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { appPrefix } from "../api/client";
import type { DatasetInspection, EventError, HubJob } from "../api/types";
import { t, tpOptional } from "../i18n";
import { hubError, isRepoId, pollHubJob, recentRepos, rememberRepo } from "../lib/hubJob";
import { runErrorText } from "../lib/runWarning";
import { untrustedText } from "../lib/safeText";
import { useWorkflow } from "../store/workflow";
import { ErrorIcon } from "./ErrorIcon";
import { FieldError } from "./FieldError";
import { Icon } from "./Icon";

/** A sample's error code (backend H4) in the user's words; an unknown code reads as "unreadable", never raw. */
function sampleErrorText(code: string): string {
  return tpOptional(["byo", "sample_error", code]) ?? t("byo.sample_error.image_unreadable");
}

type Mapping = { image?: string; question?: string; answer?: string; messages?: string };

/**
 * "Ghép cột dữ liệu" (spec H §4): which column is the image, the question, the answer (or the conversation), the
 * server's suggestions pre-filled, a wrong kind said under its select, and 3-5 samples previewed. Sample text is the
 * repo's own (untrusted): plain text, clamped.
 */
function ColumnMapDialog({
  inspection,
  inspectId,
  onApply,
  onClose,
}: {
  inspection: DatasetInspection;
  inspectId: string;
  onApply: (m: Mapping, fixedQuestion: string) => void;
  onClose: () => void;
}) {
  const s = inspection.suggested;
  const [mode, setMode] = useState<"qa" | "messages">(s.messages && !s.answer ? "messages" : "qa");
  const [map, setMap] = useState<Mapping>(s);
  const [fixed, setFixed] = useState("");
  const kindOf = (name?: string) => inspection.columns.find((c) => c.name === name)?.kind;
  const roles =
    mode === "qa" ? (["image", "question", "answer"] as const) : (["messages", "image"] as const);
  const want: Record<string, string> = {
    image: "image",
    question: "text",
    answer: "text",
    messages: "messages",
  };
  const wrong = (role: string) => {
    const col = map[role as keyof Mapping];
    return !!col && kindOf(col) !== want[role];
  };
  const missing =
    mode === "qa" ? !map.image || !map.answer || (!map.question && !fixed.trim()) : !map.messages;
  const ok = !missing && !roles.some(wrong);

  return (
    <div className="modal-scrim">
      <div
        className="modal byo-map"
        role="dialog"
        aria-modal="true"
        aria-labelledby="byo-map-title"
      >
        <h2 id="byo-map-title">{t("byo.map.title")}</h2>
        <fieldset className="byo-map-mode">
          <legend className="sr-only">{t("byo.map.title")}</legend>
          {(["qa", "messages"] as const).map((m) => (
            <label key={m} className="field check">
              <input
                type="radio"
                name="byo-map-mode"
                checked={mode === m}
                onChange={() => setMode(m)}
              />
              {t(m === "qa" ? "byo.map.mode_qa" : "byo.map.mode_messages")}
            </label>
          ))}
        </fieldset>
        {roles.map((role) => (
          <div key={role} className="byo-map-row">
            <p className="byo-map-head">
              <label htmlFor={`byo-col-${role}`}>{t(`byo.map.${role}`)}</label>
              {s[role] && s[role] === map[role] && (
                <span className="muted"> · {t("byo.map.suggested")}</span>
              )}
            </p>
            <select
              id={`byo-col-${role}`}
              value={map[role] ?? ""}
              onChange={(e) => setMap({ ...map, [role]: e.target.value || undefined })}
            >
              <option value="">—</option>
              {inspection.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {untrustedText(c.name, 60)}
                </option>
              ))}
            </select>
            {wrong(role) && (
              <FieldError>
                {
                  runErrorText(
                    { code: "DATA-011", params: { column: map[role] ?? "", role } },
                    (x) => x,
                  ).title
                }
              </FieldError>
            )}
          </div>
        ))}
        {mode === "qa" && !map.question && (
          <label className="byo-map-row">
            {t("byo.map.fixed_question")}
            <input type="text" value={fixed} onChange={(e) => setFixed(e.target.value)} />
          </label>
        )}
        <p className="small muted">{t("byo.map.preview")}</p>
        {inspection.sample_error && (
          <p className="small" role="status">
            {t(`byo.sample_error.${inspection.sample_error}`)}
          </p>
        )}
        <ul className="byo-preview">
          {inspection.samples.slice(0, 5).map((smp, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: a fixed list of samples
            <li key={i}>
              {smp.error ? (
                <span className="byo-thumb err" aria-hidden="true">
                  <Icon name="triangle-alert" size={16} />
                </span>
              ) : smp.thumb_url ? (
                <img
                  className="byo-thumb"
                  src={`${appPrefix()}${smp.thumb_url}`}
                  alt=""
                  width={64}
                  height={64}
                  loading="lazy"
                />
              ) : (
                <span className="byo-thumb" aria-hidden="true" />
              )}
              <span className="byo-sample">
                {smp.error ? (
                  <span className="small">{sampleErrorText(smp.error)}</span>
                ) : (
                  <>
                    <span className="clamp2">{untrustedText(smp.question ?? fixed, 300)}</span>
                    <span className="clamp2 muted mono">
                      {untrustedText(smp.answer ?? "", 300)}
                    </span>
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
        <p className="modal-actions">
          <button type="button" className="hairline" onClick={onClose}>
            {t("storage.dialog.cancel")}
          </button>
          <button
            type="button"
            className="primary"
            disabled={!ok}
            aria-disabled={!ok}
            data-inspect={inspectId}
            onClick={() =>
              onApply(
                mode === "qa"
                  ? { image: map.image, question: map.question, answer: map.answer }
                  : { messages: map.messages, image: map.image },
                fixed,
              )
            }
          >
            {t("byo.map.apply")}
          </button>
        </p>
      </div>
    </div>
  );
}

/**
 * "Từ Hugging Face" for a dataset step (ADR-021 §4, spec H §4): paste a repo id, pick config/split when there are
 * several, check, then map its columns; the mapping lands in the step's params and the usual data checks run on
 * validate (in the check bar, not in the dialog).
 */
export function DatasetHubPanel({ nodeId }: { nodeId: string }) {
  const [repo, setRepo] = useState("");
  const [config, setConfig] = useState("");
  const [split, setSplit] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<EventError | null>(null);
  const [found, setFound] = useState<{ id: string; result: DatasetInspection } | null>(null);
  const [mapping, setMapping] = useState(false);
  const alive = useRef(true);
  // set on every mount: StrictMode mounts twice in dev, and a ref left false would stop the poll for good
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const recent = recentRepos("dataset");

  const check = async (value = repo) => {
    const r = value.trim();
    if (!isRepoId(r) || busy) return;
    setRepo(r);
    setBusy(true);
    setError(null);
    try {
      const { inspect_id } = await api.inspectDataset({
        repo: r,
        ...(config ? { config } : {}),
        ...(split ? { split } : {}),
      });
      rememberRepo("dataset", r);
      const end = await pollHubJob<HubJob<DatasetInspection>>(
        () => api.datasetInspection(inspect_id),
        () => undefined,
        () => !alive.current,
      );
      if (!alive.current || !end) return;
      if (end.status === "failed" && end.error) setError(end.error);
      else if (end.result) {
        setFound({ id: inspect_id, result: end.result });
        setMapping(true);
      }
    } catch (e) {
      if (alive.current) setError(hubError(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const apply = (m: Mapping, fixedQuestion: string) => {
    if (!found) return;
    const r = found.result;
    // one edit, one undo step
    useWorkflow.getState().setParams(nodeId, {
      repo: r.repo,
      revision: r.revision,
      ...(r.config ? { config: r.config } : {}),
      split: r.split,
      columns: {
        ...Object.fromEntries(Object.entries(m).filter(([, v]) => v)),
        ...(fixedQuestion.trim() ? { fixed_question: fixedQuestion.trim() } : {}),
      },
    });
    setMapping(false);
  };

  const errText = error ? runErrorText(error, (id) => id) : undefined;
  const choices = found?.result;
  return (
    <div className="byo-panel">
      <div className="byo-input">
        <input
          type="text"
          className="mono"
          spellCheck={false}
          autoComplete="off"
          aria-label={t("byo.source.hub")}
          placeholder={t("byo.data.placeholder")}
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
      {choices && choices.configs.length > 1 && (
        <label className="byo-map-row">
          {t("byo.data.config")}
          <select
            value={config || choices.config || ""}
            onChange={(e) => setConfig(e.target.value)}
          >
            {choices.configs.map((c) => (
              <option key={c}>{untrustedText(c, 60)}</option>
            ))}
          </select>
        </label>
      )}
      {choices && choices.splits.length > 1 && (
        <label className="byo-map-row">
          {t("byo.data.split")}
          <select value={split || choices.split} onChange={(e) => setSplit(e.target.value)}>
            {choices.splits.map((c) => (
              <option key={c}>{untrustedText(c, 60)}</option>
            ))}
          </select>
        </label>
      )}
      {recent.length > 0 && !busy && !found && (
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
      {found && !mapping && (
        <p className="small">
          <code>{found.result.repo}</code> ·{" "}
          <button type="button" className="link" onClick={() => setMapping(true)}>
            {t("byo.map.title")}
          </button>
        </p>
      )}
      {found && mapping && (
        <ColumnMapDialog
          inspection={found.result}
          inspectId={found.id}
          onApply={apply}
          onClose={() => setMapping(false)}
        />
      )}
    </div>
  );
}
