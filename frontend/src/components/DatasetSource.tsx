import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { appPrefix } from "../api/client";
import type { DatasetInspection, EventError, HubJob, UserDataset } from "../api/types";
import { t, tpOptional } from "../i18n";
import {
  configSplitOf,
  hubError,
  isRepoId,
  pollHubJob,
  recentRepos,
  rememberRepo,
} from "../lib/hubJob";
import { runErrorText } from "../lib/runWarning";
import { untrustedText } from "../lib/safeText";
import { useUserModels } from "../store/userModels";
import { useWorkflow } from "../store/workflow";
import { ErrorIcon } from "./ErrorIcon";
import { FieldError } from "./FieldError";
import { Icon } from "./Icon";

/** A sample's error code (backend H4) in the user's words; an unknown code reads as "unreadable", never raw. */
function sampleErrorText(code: string): string {
  return tpOptional(["byo", "sample_error", code]) ?? t("byo.sample_error.image_unreadable");
}

type Mapping = {
  image?: string;
  question?: string;
  answer?: string;
  messages?: string;
  /** The question every sample without one gets (text, not a column; backend ColumnMap.prompt, ≤ 2000). */
  prompt?: string;
};
/** backend ColumnMap.prompt limit (above it: 400 CFG-005 range) */
const PROMPT_MAX = 2000;

/** The three params a user-added dataset sets together (backend H5); the server refuses them when they disagree. */
export const USER_DATASET_PARAMS = ["user_dataset", "user_repo", "user_revision"];

/** Pick a user-added dataset for the step (id, repo, pinned revision), or `null` back to the catalog one: one undo step. */
export function setUserDataset(
  nodeId: string,
  ds: Pick<UserDataset, "id" | "repo" | "revision"> | null,
) {
  useWorkflow.getState().setParams(nodeId, {
    user_dataset: ds?.id ?? "",
    user_repo: ds?.repo ?? "",
    user_revision: ds?.revision ?? "",
  });
}

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
  /** Adds the dataset on the server; resolves to its refusal (DATA-010…) or null once added. */
  onApply: (m: Mapping) => Promise<EventError | null>;
  onClose: () => void;
}) {
  const s = inspection.suggested;
  const [mode, setMode] = useState<"qa" | "messages">(s.messages && !s.answer ? "messages" : "qa");
  const [map, setMap] = useState<Mapping>(s);
  const [busy, setBusy] = useState(false);
  const [refused, setRefused] = useState<EventError | null>(null);
  // the shared question (design r3): the server's default typed in, editable; sent only when it differs
  const defaultQ = inspection.default_question ?? "";
  const [prompt, setPrompt] = useState(defaultQ);
  const askPrompt = mode === "qa" && !map.question;
  const promptIsDefault = prompt.trim() === defaultQ.trim();
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
  // the server's rule (ColumnMap): an image, and an answer (question optional) or a conversation
  const missing = !map.image || (mode === "qa" ? !map.answer : !map.messages);
  const ok = !missing && !roles.some(wrong) && !busy;
  const apply = async () => {
    setBusy(true);
    setRefused(null);
    const err = await onApply(
      mode === "qa"
        ? {
            image: map.image,
            question: map.question,
            answer: map.answer,
            ...(askPrompt && prompt.trim() && !promptIsDefault ? { prompt: prompt.trim() } : {}),
          }
        : { messages: map.messages, image: map.image },
    );
    setBusy(false);
    if (err) setRefused(err);
  };
  const refusedText = refused ? runErrorText(refused, (x) => x) : undefined;

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
              aria-invalid={wrong(role) || undefined}
              aria-describedby={wrong(role) ? `byo-col-${role}-err` : undefined}
              value={map[role] ?? ""}
              onChange={(e) => setMap({ ...map, [role]: e.target.value || undefined })}
            >
              <option value="">{tpOptional(["byo", "map", "none"]) ?? "—"}</option>
              {inspection.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {untrustedText(c.name, 60)}
                </option>
              ))}
            </select>
            {wrong(role) && (
              <FieldError id={`byo-col-${role}-err`}>
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
        {askPrompt && (
          <div className="byo-map-row">
            <p className="byo-map-head">
              <label htmlFor="byo-prompt">{t("byo.map.prompt")}</label>
              <button
                type="button"
                className="help-btn inline"
                popoverTarget="byo-prompt-help"
                aria-label={t("d4.explain")}
              >
                <Icon name="circle-question-mark" size={14} />
              </button>
            </p>
            <div id="byo-prompt-help" popover="auto" className="pop">
              <strong>{t("byo.map.prompt")}</strong>
              <p>{t("byo.map.prompt_help")}</p>
            </div>
            <input
              id="byo-prompt"
              type="text"
              maxLength={PROMPT_MAX}
              value={prompt}
              aria-describedby={promptIsDefault ? "byo-prompt-note" : undefined}
              onChange={(e) => setPrompt(e.target.value)}
            />
            {promptIsDefault && (
              <p id="byo-prompt-note" className="small muted">
                {t("byo.map.prompt_default_note")}
              </p>
            )}
          </div>
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
                    {!askPrompt ? (
                      <span className="clamp2">{untrustedText(smp.question ?? "", 300)}</span>
                    ) : (
                      // no question column: the one every sample gets (the default reads muted/italic, the user's own
                      // question plain)
                      <span className={`clamp2${promptIsDefault ? " byo-default-q" : ""}`}>
                        {untrustedText(prompt.trim() || defaultQ, 300)}
                      </span>
                    )}
                    <span className="clamp2 muted mono">
                      {untrustedText(smp.answer ?? "", 300)}
                    </span>
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
        {refused && (
          <div className="error-card" role="alert" data-code={refused.code}>
            <ErrorIcon tone="critical" />
            <strong>{refusedText?.title}</strong>
            {refusedText?.what && <p>{refusedText.what}</p>}
          </div>
        )}
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
            onClick={() => void apply()}
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
  const [revision, setRevision] = useState("");
  const [revisionOpen, setRevisionOpen] = useState(false);
  // the dataset this step uses now (this machine's record), said above the paste box
  const usedId = useWorkflow((st) => {
    const v = st.workflow?.nodes.find((n) => n.id === nodeId)?.params.user_dataset;
    return typeof v === "string" ? v : "";
  });
  const inUse = useUserModels((st) => st.datasets.find((d) => d.id === usedId));
  // DATA-013 "Thêm dataset này": the workflow's repo + pinned revision typed in, never checked for the user
  useEffect(() => {
    const p = useUserModels.getState().takePrefill(nodeId);
    if (p) {
      setRepo(p.repo);
      setRevision(p.revision);
      setRevisionOpen(!!p.revision);
    }
    void useUserModels.getState().load();
  }, [nodeId]);

  const check = async (value = repo, pick?: { config: string; split: string }) => {
    const r = value.trim();
    if (!isRepoId(r) || (busy && !pick)) return;
    setRepo(r);
    setBusy(true);
    setError(null);
    try {
      const { inspect_id } = await api.inspectDataset({
        repo: r,
        ...(revision.trim() ? { revision: revision.trim() } : {}),
        ...((pick?.config ?? config) ? { config: pick?.config ?? config } : {}),
        ...((pick?.split ?? split) ? { split: pick?.split ?? split } : {}),
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
        // the step already names a user dataset (DATA-013, another machine): the same config/split, checked once
        const same =
          !pick && usedId ? configSplitOf(usedId, end.result.configs, end.result.splits) : null;
        if (
          same &&
          (same.config !== (end.result.config ?? "default") || same.split !== end.result.split)
        ) {
          setConfig(same.config);
          setSplit(same.split);
          await check(r, same);
          return;
        }
        setFound({ id: inspect_id, result: end.result });
        setMapping(true);
      }
    } catch (e) {
      if (alive.current) setError(hubError(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const apply = async (m: Mapping): Promise<EventError | null> => {
    if (!found) return null;
    try {
      const columns = Object.fromEntries(
        Object.entries(m).filter((e): e is [string, string] => !!e[1]),
      );
      const { dataset } = await api.addUserDataset({ inspect_id: found.id, columns });
      setUserDataset(nodeId, dataset);
      void useUserModels.getState().load();
      if (alive.current) setMapping(false);
      return null;
    } catch (e) {
      return hubError(e);
    }
  };

  const splitSelect = !!found && found.result.splits.length > 1;
  const summary = inUse
    ? { repo: inUse.repo, config: inUse.config ?? undefined, split: inUse.split }
    : found
      ? {
          repo: found.result.repo,
          config: found.result.config ?? undefined,
          split: found.result.split,
        }
      : null;
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
      <details
        className="byo-revision"
        open={revisionOpen}
        onToggle={(e) => setRevisionOpen(e.currentTarget.open)}
      >
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
      {summary && !mapping && (
        // one block (design H r2): repo + chip, "config · split" (the split only when there is no split select),
        // then "Ghép cột dữ liệu"
        <div className="byo-ds-summary">
          <p className="small">
            <code>{untrustedText(summary.repo, 120)}</code>{" "}
            {inUse && (
              <span className="chip unverified">
                <Icon name="badge-question-mark" size={12} /> {t("byo.unverified")}
              </span>
            )}
          </p>
          <p className="small muted">
            {[summary.config, splitSelect ? null : summary.split]
              .filter((x): x is string => !!x)
              .map((x) => untrustedText(x, 60))
              .join(" · ")}
          </p>
          {found && (
            <p className="small">
              <button type="button" className="link" onClick={() => setMapping(true)}>
                {t("byo.map.title")}
              </button>
            </p>
          )}
        </div>
      )}
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
      {splitSelect && choices && (
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
