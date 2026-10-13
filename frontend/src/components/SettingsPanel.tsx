import { useEffect, useState } from "react";
import { api } from "../api";
import { ApiError } from "../api/client";
import type { SecretField, SecretState, SettingsView } from "../api/types";
import { t, tpOptional } from "../i18n";
import { formatDateTime } from "../lib/format";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";

const FIELDS: SecretField[] = ["HF_TOKEN", "NGC_API_KEY", "WANDB_API_KEY"];

/** Server error → copy (settings_error.<code>); params fill the sentence; unknown codes get a generic line. */
function errorText(e: unknown): string {
  const err =
    e instanceof ApiError
      ? (e.body as { error?: { code?: string; params?: Record<string, string> } } | null)?.error
      : undefined;
  if (err?.code === "settings_invalid" && err.params?.reason) {
    const why = tpOptional(["settings_error", "invalid", err.params.reason]);
    if (why) return why;
  }
  return (
    (err?.code && tpOptional(["settings_error", err.code], err.params)) ||
    t("settings_error.generic")
  );
}

/**
 * One secret (D-7). The value is write-only: the input always starts empty, is cleared right after saving and is
 * never kept in state beyond the request; the row shows only "set …ab12 · updated <date>" (PM/security rules).
 */
function SecretRow({
  field,
  state,
  onChange,
}: {
  field: SecretField;
  state: SecretState;
  onChange: (s: SecretState) => void;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const id = `secret-${field}`;

  const save = async () => {
    setBusy(true);
    setError(null);
    const v = value;
    setValue(""); // never keep the token around, even if the request fails
    try {
      onChange(await api.putSecret(field, v));
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.deleteSecret(field);
      onChange({ set: false, updated_at: null, last4: "", readable: true, scope: "unchecked" });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div className="set-row" data-field={field}>
      <span className="set-label">
        <label htmlFor={id}>{t(`settings.secret.${field}`)}</label>
        <button
          type="button"
          className="help-btn inline"
          popoverTarget={`help-${field}`}
          aria-label={t("d4.explain")}
        >
          <Icon name="circle-question-mark" size={14} />
        </button>
        <span id={`help-${field}`} popover="auto" className="pop">
          <span className="pop-p">{t(`settings.secret_help.${field}`)}</span>
          <span className="pop-p">{t("settings.no_password_manager")}</span>
        </span>
      </span>
      <div className="set-control">
        <p className="set-state">
          {state.set && state.readable ? (
            <>
              <span className="si si-completed">
                <Icon name="circle-check-big" size={14} />
              </span>{" "}
              {t("settings.secret_set", { last4: state.last4 })}
              {state.updated_at
                ? ` · ${t("settings.updated", { time: formatDateTime(state.updated_at * 1000) })}`
                : ""}
            </>
          ) : state.set ? (
            <>
              <span className="si si-warn">
                <Icon name="triangle-alert" size={14} />
              </span>{" "}
              {t("settings.secret_unreadable")}
            </>
          ) : (
            <span className="muted">{t("settings.secret_unset")}</span>
          )}
        </p>
        {state.set && state.scope === "unchecked" && (
          <p className="muted small">{t("settings.scope_unchecked")}</p>
        )}
        <div className="set-input">
          <input
            id={id}
            type="password"
            // keep tokens out of browser password managers (security D-7): no <form>, plain button, no username
            autoComplete="new-password"
            data-1p-ignore=""
            data-lpignore="true"
            spellCheck={false}
            value={value}
            placeholder={
              state.set ? t("settings.replace_placeholder") : t("settings.enter_placeholder")
            }
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && value) void save();
            }}
          />
          <button
            type="button"
            className="btn primary sm"
            disabled={!value || busy}
            aria-disabled={!value || busy}
            onClick={() => void save()}
          >
            {t("settings.save")}
          </button>
          {state.set && !confirmDelete && (
            <button
              type="button"
              className="btn ghost sm"
              disabled={busy}
              onClick={() => setConfirmDelete(true)}
            >
              {t("settings.delete")}
            </button>
          )}
          {confirmDelete && (
            <span className="set-confirm" role="alert">
              {t("settings.delete_confirm")}{" "}
              <button type="button" className="btn ghost sm" onClick={() => void remove()}>
                {t("settings.delete")}
              </button>{" "}
              <button type="button" className="link" onClick={() => setConfirmDelete(false)}>
                {t("storage.dialog.cancel")}
              </button>
            </span>
          )}
        </div>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

/** Policy value for people: its copy (settings.policy_value.<key>.<value>), "N GB" for *_gb, else as written. */
const show = (key: string, v: unknown): { text: string; mono: boolean } => {
  if (Array.isArray(v))
    return { text: v.length ? v.join(", ") : t("settings.none"), mono: v.length > 0 };
  if (v === null || v === undefined) return { text: "—", mono: false };
  const label = tpOptional(["settings", "policy_value", key, String(v)]);
  if (label) return { text: label, mono: false };
  if (key.endsWith("_gb") && typeof v === "number") return { text: `${v} GB`, mono: true };
  return { text: String(v), mono: true };
};

/** Settings (D4, D-7): secrets the user may set; deployment policy shown read-only with who/why behind "?". */
export function SettingsPanel() {
  const [view, setView] = useState<SettingsView | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    api.settings().then(setView, () => setFailed(true));
  }, []);
  if (failed) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!view) return <div className="skeleton-grid" aria-busy="true" />;

  return (
    <main className="settings-panel">
      <h1>{t("settings.title")}</h1>
      {!view.key_ok && (
        <div className="error-card" role="alert">
          <ErrorIcon tone="critical" />
          <strong>{t("settings.key_broken_title")}</strong>
          <p>{t("settings.key_broken_what")}</p>
        </div>
      )}
      <section aria-labelledby="set-connect">
        <h2 id="set-connect">
          <Icon name="key-round" size={16} /> {t("settings.section_connect")}
        </h2>
        {FIELDS.map((f) => (
          <SecretRow
            key={f}
            field={f}
            state={view.secrets[f]}
            onChange={(s) =>
              setView((cur) => (cur ? { ...cur, secrets: { ...cur.secrets, [f]: s } } : cur))
            }
          />
        ))}
      </section>
      <section aria-labelledby="set-admin">
        <h2 id="set-admin">
          <Icon name="lock" size={16} /> {t("settings.section_admin")}
        </h2>
        {Object.entries(view.admin_policy).map(([key, { value, source, reason_key, warning }]) => {
          const popId = `pol-${key.replace(/\W+/g, "-")}`;
          // design: under its row, warning icon in status ink, text in text tokens (Ink-Not-Label), no page banner
          const warn = warning ? tpOptional(["settings", "policy_warning", warning]) : undefined;
          return (
            <div key={key} className="set-row locked" data-key={key}>
              <span className="set-label">{tpOptional(["settings", "policy", key]) ?? key}</span>
              <div className="set-control">
                {source === "default" ? (
                  <span className="muted">{t("settings.policy_default")}</span>
                ) : (
                  <span className={show(key, value).mono ? "mono" : undefined}>
                    {show(key, value).text}
                  </span>
                )}
                <button
                  type="button"
                  className="help-btn inline"
                  popoverTarget={popId}
                  aria-label={t("settings.locked_why")}
                >
                  <Icon name="lock" size={14} />
                </button>
                <div id={popId} popover="auto" className="pop">
                  <strong>{t("settings.locked_why")}</strong>
                  <p>
                    {(reason_key && tpOptional(reason_key.split("."))) ||
                      (source === "default"
                        ? t("settings.locked_default")
                        : t("settings.locked_by_admin"))}
                  </p>
                </div>
              </div>
              {warn && (
                <p className="set-warning" role="status">
                  <span className="si si-warn">
                    <Icon name="triangle-alert" size={14} />
                  </span>{" "}
                  {warn}
                </p>
              )}
            </div>
          );
        })}
      </section>
    </main>
  );
}
