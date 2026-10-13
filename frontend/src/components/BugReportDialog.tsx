import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { BugReport } from "../api/types";
import { getLocale, t } from "../i18n";
import { Icon } from "./Icon";
import { toast } from "./QuickFixes";

/** The project's public issue tracker (PM 2026-10-13). Nothing is prefilled in the URL: the text goes by clipboard. */
export const ISSUES_URL = "https://github.com/baonguyen3568/training-line/issues/new";

const FIELDS = ["code", "error_title", "app_version", "gpu", "ref_code"] as const;

/**
 * Plain text of the report: one "Label: value" line per field the server sent (absent fields are left out), then
 * the security reminder as the last line (security condition 2).
 */
export function reportText(r: BugReport): string {
  return [
    ...FIELDS.filter((f) => r[f]).map((f) => `${t(`bug_report.field.${f}`)}: ${r[f]}`),
    "",
    t("bug_report.reminder"),
  ].join("\n");
}

/**
 * "Gửi báo cáo lỗi" (design ba32c2c): shows exactly what will be copied — five safe fields from the server, never
 * logs, paths, tokens or data — then copies it and opens the project's GitHub issue page. The app sends nothing.
 */
export function BugReportDialog({
  runId,
  nodeId,
  code,
  onClose,
}: {
  runId: string;
  nodeId?: string;
  code?: string;
  onClose: () => void;
}) {
  const [report, setReport] = useState<BugReport | null>(null);
  const [failed, setFailed] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const pre = useRef<HTMLPreElement>(null);
  const opener = useRef<Element | null>(document.activeElement);

  useEffect(() => {
    api.bugReport(runId, { nodeId, code }, getLocale()).then(setReport, () => setFailed(true));
  }, [runId, nodeId, code]);

  const close = useCallback(() => {
    onClose();
    (opener.current as HTMLElement | null)?.focus?.();
  }, [onClose]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  const text = report ? reportText(report) : "";
  const copyOpen = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // no clipboard (permissions, http): let the user copy by hand, text already selected
      setCopyFailed(true);
      const sel = window.getSelection();
      if (pre.current && sel) {
        const range = document.createRange();
        range.selectNodeContents(pre.current);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      return;
    }
    window.open(ISSUES_URL, "_blank", "noopener,noreferrer");
    toast(t("bug_report.copied"));
    close();
  };

  return (
    <div className="modal-scrim">
      <div
        className="modal bug-report"
        role="dialog"
        aria-modal="true"
        aria-labelledby="bug-report-title"
      >
        <h2 id="bug-report-title">
          <Icon name="message-square" size={16} /> {t("bug_report.title")}
        </h2>
        <p>{t("bug_report.intro")}</p>
        {failed && <p role="alert">{t("app.catalog_load_failed")}</p>}
        {!report && !failed && <div className="skeleton-grid" aria-busy="true" />}
        {report && (
          <>
            <p className="small">
              {t("bug_report.preview_label")}
              {report.ref_code && (
                <>
                  <button
                    type="button"
                    className="help-btn inline"
                    popoverTarget="ref-code-help"
                    aria-label={t("bug_report.field.ref_code")}
                  >
                    <Icon name="circle-question-mark" size={14} />
                  </button>
                  <span id="ref-code-help" popover="auto" className="pop">
                    {t("bug_report.ref_code_help")}
                  </span>
                </>
              )}
            </p>
            <pre className="bug-report-preview" ref={pre}>
              {text}
            </pre>
            <p className="muted small">{t("bug_report.not_included")}</p>
            <p className="muted small">{t("bug_report.needs_account")}</p>
          </>
        )}
        {copyFailed && (
          <p className="banner" role="status">
            {t("bug_report.copy_failed")}
          </p>
        )}
        <p className="modal-actions">
          <button type="button" className="hairline" onClick={close}>
            {t("storage.dialog.cancel")}
          </button>
          {report && (
            <button type="button" className="primary" onClick={() => void copyOpen()}>
              <Icon name="external-link" size={14} /> {t("bug_report.copy_open")}
            </button>
          )}
        </p>
      </div>
    </div>
  );
}
