import { useState } from "react";
import { api } from "../api";
import { ApiError } from "../api/client";
import type { SecretField, SecretState } from "../api/types";
import { t, tpOptional } from "../i18n";
import { formatDateTime } from "../lib/format";
import { FieldError } from "./FieldError";
import { Icon } from "./Icon";

// Its own module so the editor (the gated download steps) can use the row without pulling the lazy Settings page.

/** Server error → copy (settings_error.<code>); params fill the sentence; unknown codes get a generic line. */
export function errorText(e: unknown): string {
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
export function SecretRow({
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
        {error && <FieldError>{error}</FieldError>}
      </div>
    </div>
  );
}
