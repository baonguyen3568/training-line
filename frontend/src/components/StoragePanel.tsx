import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { ApiError } from "../api/client";
import type {
  RenderedError,
  StorageApplyResult,
  StorageItem,
  StoragePlan,
  StorageSection,
} from "../api/types";
import { getLocale, t, tpOptional } from "../i18n";
import { formatDateTime } from "../lib/format";
import { untrustedText } from "../lib/safeText";

const SECTIONS: StorageSection[] = ["manual", "auto", "kept"];
type Params = Record<string, string | number>;

/** Copy with every placeholder filled, or undefined (never show "{minutes}" to the user). */
function filled(path: string[], params: Params): string | undefined {
  const text = tpOptional(path, params);
  return text === undefined || /\{\w+\}/.test(text) ? undefined : text;
}

function itemLabel(item: StorageItem): string {
  if (item.display_name) return untrustedText(item.display_name, 80);
  return filled(item.display_key.split("."), item.display_params) ?? item.item_id;
}

function consequence(item: StorageItem): string | undefined {
  if (!item.consequence_code) return undefined;
  const path = ["storage", "consequence", item.consequence_code];
  const p = item.consequence_params;
  return (
    filled(path, p) ??
    filled([...path, "workflow" in p ? "default" : "no_workflow"], p) ??
    undefined
  );
}

function reason(item: StorageItem): string | undefined {
  if (item.lock_reason)
    return filled(["storage", "lock_reason", item.lock_reason], item.lock_params ?? {});
  if (item.protected_reason)
    return filled(["storage", "protected_reason", item.protected_reason], {});
  return undefined;
}

const gb = (v: number | null | undefined) => (v === null || v === undefined ? "—" : v.toFixed(1));

type Outcome =
  | { kind: "done"; result: StorageApplyResult }
  | { kind: "busy" }
  | { kind: "approval"; error: RenderedError };

/** Storage panel (P013 G-1): one plan over node cache, checkpoints and set-aside runs; nothing is deleted by path. */
export function StoragePanel() {
  const [plan, setPlan] = useState<StoragePlan | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState<Record<string, string>>({});
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = useCallback(() => {
    setLoadError(false);
    api.storagePlan(getLocale()).then(
      (p) => {
        setPlan(p);
        setSelected(
          (cur) => new Set([...cur].filter((id) => p.items.some((i) => i.item_id === id))),
        );
      },
      () => setLoadError(true),
    );
  }, []);
  useEffect(load, [load]);

  if (loadError) return <p role="alert">{t("app.catalog_load_failed")}</p>;
  if (!plan) return <div className="skeleton-grid" aria-busy="true" />;

  const chosen = plan.items.filter((i) => selected.has(i.item_id));
  const needTyping = chosen.filter((i) => i.confirm_required);
  const allTyped = needTyping.every((i) => typed[i.item_id] === i.confirm_text);
  const chosenGb = chosen.reduce((sum, i) => sum + (i.size_gb ?? 0), 0);
  const keptTotal = plan.items
    .filter((i) => i.section === "kept")
    .reduce((sum, i) => sum + (i.size_gb ?? 0), 0);

  const toggle = (id: string) =>
    setSelected((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const apply = async () => {
    setBusy(true);
    setNotice(null);
    try {
      const confirm: Record<string, string> = {};
      for (const i of needTyping) confirm[i.item_id] = typed[i.item_id] ?? "";
      const result = await api.storageApply(
        { plan_id: plan.plan_id, item_ids: chosen.map((i) => i.item_id), confirm },
        getLocale(),
      );
      setOutcome({ kind: "done", result });
      setSelected(new Set());
      setTyped({});
      setConfirming(false);
      load();
    } catch (e) {
      const data = e instanceof ApiError ? (e.body as Record<string, unknown> | null) : null;
      if (e instanceof ApiError && e.status === 409 && data?.code === "PLAN_EXPIRED") {
        setNotice(t("storage.dialog.plan_expired"));
        load(); // keep the selection that still exists; the user confirms again on the fresh plan
      } else if (
        e instanceof ApiError &&
        e.status === 409 &&
        data?.code === "CLEANUP_IN_PROGRESS"
      ) {
        setOutcome({ kind: "busy" }); // not an error, and the plan is still valid
      } else if (e instanceof ApiError && e.status === 403 && data?.error) {
        setOutcome({ kind: "approval", error: data.error as RenderedError });
        setConfirming(false);
      } else {
        setNotice(t("app.catalog_load_failed"));
      }
    } finally {
      setBusy(false);
    }
  };

  const adminCommand =
    outcome?.kind === "approval" ? String(outcome.error.params?.admin_command ?? "") : "";

  return (
    <main className="gallery storage-panel">
      <header className="storage-head">
        <h1>{t("storage.panel.title")}</h1>
        <button type="button" className="link" onClick={load}>
          ↻ {t("storage.panel.refresh")}
        </button>
      </header>

      {plan.error && (
        <div className="error-card warn" role="status" data-code={plan.error.code}>
          <strong>{plan.error.title}</strong>
          <p>{plan.error.what}</p>
        </div>
      )}
      {notice && <p className="banner">{notice}</p>}
      {outcome?.kind === "busy" && (
        <p className="banner" role="status">
          ⏳ {t("storage.panel.cleanup_in_progress")}
        </p>
      )}
      {outcome?.kind === "done" && (
        <div className="banner" role="status" data-testid="storage-result">
          ✓ {t("storage.panel.freed", { gb: gb(outcome.result.freed_gb) })}
          {outcome.result.skipped.length > 0 && (
            <>
              <p>
                {t("storage.panel.skipped_title", { item_count: outcome.result.skipped.length })}
              </p>
              <ul>
                {outcome.result.skipped.map((s) => {
                  const item = plan.items.find((i) => i.item_id === s.item_id);
                  return (
                    <li key={s.item_id}>
                      {item ? itemLabel(item) : untrustedText(s.item_id, 80)}:{" "}
                      {filled(["storage", "skip_reason", s.reason], {}) ??
                        filled(["storage", "lock_reason", s.reason], {}) ??
                        filled(["storage", "protected_reason", s.reason], {}) ??
                        t("storage.skip_reason.NOT_APPLICABLE")}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}
      {outcome?.kind === "approval" && (
        <div className="error-card warn" role="alert" data-code={outcome.error.code}>
          <strong>{outcome.error.title}</strong>
          <p>{outcome.error.what}</p>
          {adminCommand && (
            <>
              <p className="muted small">{t("storage.panel.admin_command")}</p>
              <pre className="admin-command">
                <code>{adminCommand}</code>
              </pre>
              <button
                type="button"
                className="link"
                onClick={() =>
                  navigator.clipboard?.writeText(adminCommand).then(
                    () => setCopied(true),
                    () => {},
                  )
                }
              >
                {copied ? `✓ ${t("storage.panel.copied")}` : t("storage.panel.copy")}
              </button>
            </>
          )}
        </div>
      )}

      {plan.items.length === 0 && <p className="muted">{t("storage.panel.empty")}</p>}
      {SECTIONS.map((section) => {
        const items = plan.items.filter((i) => i.section === section);
        if (items.length === 0) return null;
        return (
          <section
            key={section}
            className={`storage-section ${section}`}
            aria-label={t(`storage.section.${section}`)}
          >
            <h2>{t(`storage.section.${section}`)}</h2>
            {section === "kept" && keptTotal > 0 && (
              // Where the space goes even though nothing here can be deleted (bao-cleanup: ~60 GB of pinned runs).
              <p className="muted">{t("storage.panel.kept_total", { gb: gb(keptTotal) })}</p>
            )}
            <ul className="storage-items">
              {items.map((item) => {
                const why = reason(item);
                const what = consequence(item);
                const label = itemLabel(item);
                // confirm_required without confirm_text could never pass the typed check: lock it instead (security nit).
                const selectable =
                  item.section !== "kept" &&
                  item.action === "delete" &&
                  !(item.confirm_required && !item.confirm_text);
                return (
                  <li key={item.item_id} className="storage-item" data-item={item.item_id}>
                    {selectable ? (
                      <input
                        type="checkbox"
                        aria-label={label}
                        checked={selected.has(item.item_id)}
                        onChange={() => toggle(item.item_id)}
                      />
                    ) : (
                      <span aria-hidden="true">🔒</span>
                    )}
                    <span className="storage-label">
                      <strong>{label}</strong>
                      <span className="muted small">
                        {" "}
                        {t(`storage.group.${item.group}`)}
                        {item.last_used
                          ? ` · ${t("storage.panel.last_used", { time: formatDateTime(item.last_used * 1000) })}`
                          : ""}
                      </span>
                      {why && <span className="storage-reason">{why}</span>}
                      {what && (
                        // Red is only the icon: critical text on the dark surface is 3.6:1 (< 4.5:1).
                        <span
                          className={
                            item.consequence_code === "PERMANENT_LOSS"
                              ? "consequence permanent"
                              : "muted"
                          }
                        >
                          {item.consequence_code === "PERMANENT_LOSS" && (
                            <span className="loss-icon" aria-hidden="true">
                              ⚠{" "}
                            </span>
                          )}
                          {what}
                        </span>
                      )}
                    </span>
                    <span className="storage-size">
                      {item.size_gb === null ? "—" : `${gb(item.size_gb)} GB`}
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}

      {chosen.length > 0 && !confirming && (
        <p className="storage-actions">
          <button type="button" className="primary" onClick={() => setConfirming(true)}>
            {t("storage.panel.delete_selected", { item_count: chosen.length })} · {gb(chosenGb)} GB
          </button>
        </p>
      )}
      {confirming && chosen.length > 0 && (
        <section
          role="alertdialog"
          aria-label={t("storage.dialog.title", { item_count: chosen.length, gb: gb(chosenGb) })}
          className="storage-confirm"
        >
          <h2>{t("storage.dialog.title", { item_count: chosen.length, gb: gb(chosenGb) })}</h2>
          {needTyping.map((item) => (
            <label key={item.item_id} className="typed-confirm">
              <span>
                <strong>{itemLabel(item)}</strong> — {t("storage.dialog.typed_reason_permanent")}
              </span>
              <span>
                {t("storage.dialog.typed_confirm", { confirm_text: item.confirm_text ?? "" })}
              </span>
              <input
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={typed[item.item_id] ?? ""}
                onChange={(e) => setTyped((cur) => ({ ...cur, [item.item_id]: e.target.value }))}
              />
            </label>
          ))}
          <p>
            <button type="button" className="link" onClick={() => setConfirming(false)}>
              {t("storage.dialog.cancel")}
            </button>{" "}
            <button
              type="button"
              className="primary danger"
              disabled={!allTyped || busy}
              onClick={() => void apply()}
            >
              {t("storage.dialog.confirm")}
            </button>
          </p>
        </section>
      )}
    </main>
  );
}
