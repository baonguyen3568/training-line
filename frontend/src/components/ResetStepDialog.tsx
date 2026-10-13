import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ApiError } from "../api/client";
import type { NodeResetPlan, RenderedError } from "../api/types";
import { getLocale, t, tpOptional } from "../i18n";
import { untrustedText } from "../lib/safeText";
import { fold } from "../lib/search";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";

type Params = Record<string, string | number>;
const filled = (path: string[], params: Params): string | undefined => {
  const text = tpOptional(path, params);
  return text === undefined || /\{\w+\}/.test(text) ? undefined : text;
};
const gb = (v: number) => v.toFixed(1);
/** Same loose match as the server (accents, case, repeated spaces): a hint only, the server decides. */
const same = (a: string, b: string) =>
  fold(a).replace(/\s+/g, " ").trim() === fold(b).replace(/\s+/g, " ").trim();

/**
 * G-4 "Đặt lại bước" (P013, bao-cleanup storage_reset): deletes every attempt of one graph node so it runs from
 * scratch. Read-only plan first; blocked plans say why (storage.lock_reason.*); permanent loss needs the node title
 * typed (PERMANENT_LOSS pattern of the storage panel). The plan id is single-use and expires.
 */
export function ResetStepDialog({
  node,
  onClose,
  onDone,
}: {
  node: string;
  onClose: () => void;
  onDone: (message: string) => void;
}) {
  const [plan, setPlan] = useState<NodeResetPlan | null>(null);
  const [failed, setFailed] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [approval, setApproval] = useState<RenderedError | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(document.activeElement);

  const load = useCallback(() => {
    api.nodeResetPlan(node, getLocale()).then(
      (p) => {
        setPlan(p);
        setTyped("");
      },
      () => setFailed(true),
    );
  }, [node]);
  useEffect(load, [load]);

  const close = useCallback(() => {
    onClose();
    (opener.current as HTMLElement | null)?.focus?.(); // focus goes back to what opened the dialog
  }, [onClose]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);
  useEffect(() => {
    if (plan) box.current?.querySelector<HTMLElement>("input, button.primary, button")?.focus();
  }, [plan]);

  const title = plan ? untrustedText(plan.node_title, 80) : "";
  const reason = (code: string, params: Params) =>
    filled(["storage", "lock_reason", code], params) ?? t("app.catalog_load_failed");

  const apply = async () => {
    if (!plan?.plan_id) return;
    setBusy(true);
    setNotice(null);
    try {
      const res = await api.nodeReset(
        {
          plan_id: plan.plan_id,
          node: plan.node,
          ...(plan.confirm_required ? { confirm: typed } : {}),
        },
        getLocale(),
      );
      const done = t("storage.reset_step.done", { node: title, gb: gb(res.freed_gb) });
      // 200 with failed[]: deleted in part; say so (copy from design) and let the caller reload the plan
      const partial = res.failed.length
        ? tpOptional(["storage", "reset_step", "partial"], { failed_count: res.failed.length })
        : undefined;
      onDone(partial ? `${done} ${partial}` : done);
      close();
    } catch (e) {
      const data = e instanceof ApiError ? (e.body as Record<string, unknown> | null) : null;
      const code = typeof data?.code === "string" ? data.code : "";
      if (e instanceof ApiError && e.status === 403 && data?.error) {
        setApproval(data.error as RenderedError);
      } else if (code === "CLEANUP_IN_PROGRESS") {
        setNotice(t("storage.panel.cleanup_in_progress")); // the plan is still valid: try again later
      } else if (code === "PLAN_EXPIRED" || code === "STATE_CHANGED") {
        setNotice(t("storage.dialog.plan_expired"));
        load(); // a fresh plan; the user confirms again
      } else if (code === "CONFIRM_REQUIRED") {
        setTyped("");
      } else if (code) {
        setNotice(reason(code, { node: title }));
        load();
      } else {
        setNotice(t("app.catalog_load_failed"));
      }
    } finally {
      setBusy(false);
    }
  };

  const hasCheckpoint = plan?.attempts.some((a) => a.has_checkpoint) ?? false;
  const canApply =
    !!plan?.plan_id &&
    !busy &&
    (!plan.confirm_required || (plan.confirm_text !== null && same(typed, plan.confirm_text)));

  return (
    <div className="modal-scrim">
      <div
        ref={box}
        className="modal reset-step"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="reset-step-title"
      >
        <h2 id="reset-step-title">
          <Icon name="rotate-ccw" size={16} /> {t("storage.reset_step.title")}
        </h2>
        {failed && <p role="alert">{t("app.catalog_load_failed")}</p>}
        {!plan && !failed && <div className="skeleton-grid" aria-busy="true" />}
        {plan && plan.attempts.length === 0 && <p>{t("storage.reset_step.nothing")}</p>}
        {plan && plan.attempts.length > 0 && plan.blocked_by && (
          <p className="reset-blocked" role="status">
            <span className="si si-warn">
              <Icon name="lock" size={14} />
            </span>{" "}
            {reason(plan.blocked_by, { ...plan.blocked_params, node: title })}
          </p>
        )}
        {plan && plan.attempts.length > 0 && !plan.blocked_by && (
          <>
            <p>
              {t("storage.reset_step.body", {
                attempt_count: plan.attempts.length,
                node: title,
                gb: gb(plan.total_gb),
              })}
            </p>
            <p className="muted">{t("storage.reset_step.downstream")}</p>
            <p className="reset-lost">{t("storage.reset_step.lost_lead")}</p>
            <ul className="reset-lost-list">
              {hasCheckpoint && <li>{t("storage.reset_step.lost_checkpoints")}</li>}
              <li>{t("storage.reset_step.lost_history")}</li>
            </ul>
            {plan.confirm_required && plan.confirm_text !== null && (
              <label className="typed-confirm">
                <span>{t("storage.dialog.typed_reason_permanent")}</span>
                <span>
                  {t("storage.dialog.typed_confirm", {
                    confirm_text: untrustedText(plan.confirm_text, 80),
                  })}
                </span>
                <input
                  type="text"
                  autoComplete="off"
                  spellCheck={false}
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && canApply) void apply();
                  }}
                />
              </label>
            )}
          </>
        )}
        {notice && (
          <p className="banner" role="status">
            {notice}
          </p>
        )}
        {approval && (
          <div className="error-card warn" role="alert" data-code={approval.code}>
            <ErrorIcon tone="warn" />
            <strong>{approval.title}</strong>
            <p>{approval.what}</p>
            {typeof approval.params?.admin_command === "string" && (
              <>
                <p className="muted small">{t("storage.panel.admin_command")}</p>
                <pre className="admin-command">
                  <code>{String(approval.params.admin_command)}</code>
                </pre>
              </>
            )}
          </div>
        )}
        <p className="modal-actions">
          <button type="button" className="hairline" onClick={close}>
            {t("storage.dialog.cancel")}
          </button>
          {plan?.plan_id && !plan.blocked_by && (
            <button
              type="button"
              className="primary danger"
              disabled={!canApply}
              aria-disabled={!canApply}
              onClick={() => void apply()}
            >
              {t("storage.reset_step.confirm")}
            </button>
          )}
        </p>
      </div>
    </div>
  );
}
