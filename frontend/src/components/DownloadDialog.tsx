import { useCallback, useEffect, useRef, useState } from "react";
import { t, tpOptional } from "../i18n";
import { formatDuration } from "../lib/format";
import { licenseName } from "../lib/licenses";
import { type DownloadKind, downloadKey, downloadPercent, useDownloads } from "../store/downloads";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";

export interface DownloadTarget {
  kind: DownloadKind;
  catalogId: string;
  /** What people read: the catalog label, else the id. */
  name: string;
  sizeGb?: number;
  license?: string;
  gated?: boolean;
  /** Publisher of a gated model (catalog vendor_name, else the repo's org), for the gated intro. */
  publisher?: string;
  /** Minutes the server expects the download to take (NET-002 est_min). */
  estMin?: number;
  /** NET-002 not_prepared: on disk, the server only runs the preparation step (no size, no estimate shown). */
  prepare?: boolean;
}

const gb = (bytes?: number) => (bytes === undefined ? "0" : (bytes / 1e9).toFixed(1));

/**
 * G-5 "Tải về máy" (design 701c432): what will be downloaded and how big, its licence with a required "I agree",
 * the large-download confirmation when the server asks, then progress (downloading → verifying → installing) until
 * done. The server resolves the repo from the catalog id; nothing here names a repo or a URL.
 */
export function DownloadDialog({
  target,
  onClose,
}: {
  target: DownloadTarget;
  onClose: () => void;
}) {
  const key = downloadKey(target.kind, target.catalogId);
  const entry = useDownloads((s) => s.entries[key]);
  const start = useDownloads((s) => s.start);
  const [accepted, setAccepted] = useState(false);
  const opener = useRef<Element | null>(document.activeElement);

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

  const size = entry?.sizeGb ?? target.sizeGb;
  // the server's answer comes from the pinned catalog: it wins over the client's option_meta once known
  const name = entry?.name ?? target.name;
  const license = entry?.license ?? target.license;
  const gated = entry?.gated ?? target.gated;
  const busy = entry && ["starting", "queued", "downloading", "verifying"].includes(entry.status);
  const percent = downloadPercent(entry);
  const errorTitle = entry?.error ? tpOptional(["error", entry.error.code, "title"]) : undefined;

  return (
    <div className="modal-scrim">
      <div
        className="modal download"
        role="dialog"
        aria-modal="true"
        aria-labelledby="download-title"
      >
        <h2 id="download-title">
          <Icon name="download" size={16} /> {t("download.dialog_title")}
        </h2>
        <p>
          {target.prepare
            ? t("download.intro_prepare", { name })
            : t("download.intro", {
                name,
                size_gb: size !== undefined ? size.toFixed(1) : "?",
              })}
        </p>
        {!target.prepare && target.estMin !== undefined && entry?.status !== "completed" && (
          <p className="small muted">
            {t("download.estimate", { duration: formatDuration(target.estMin * 60_000) })}
          </p>
        )}
        {gated && target.publisher && (
          // gated models need the user's own Hugging Face sign-in and terms first (the gated flow, Settings token)
          <p className="small">{t("gated.intro", { publisher: target.publisher })}</p>
        )}
        {license && (
          <p>
            {t("download.license_lead")} <strong>{licenseName(license)}</strong>
          </p>
        )}
        {!busy && entry?.status !== "completed" && entry?.status !== "unverified" && (
          <label className="field check">
            <input
              type="checkbox"
              checked={accepted}
              onChange={(e) => setAccepted(e.target.checked)}
            />
            {t("download.license_accept")}
          </label>
        )}
        {entry?.status === "needs_confirm" && (
          <p className="banner" role="status">
            {t("download.confirm_large", {
              size_gb: entry.sizeGb?.toFixed(1) ?? "?",
              threshold_gb: entry.thresholdGb ?? "?",
            })}
          </p>
        )}
        {busy && (
          <div className="download-progress" role="status" aria-live="polite">
            {entry.status === "verifying" ? (
              <p>{t("download.verifying")}</p>
            ) : (
              <>
                <progress max={100} value={percent ?? 0} aria-label={t("download.dialog_title")} />
                <p className="small">
                  {t("download.progress", {
                    done_gb: gb(entry.bytesDone),
                    total_gb: gb(entry.bytesTotal),
                    file_done_count: entry.filesDone ?? 0,
                    file_count: entry.filesTotal ?? "?",
                  })}
                </p>
              </>
            )}
          </div>
        )}
        {entry?.status === "completed" && (
          <p role="status">
            <span className="si si-completed" aria-hidden="true">
              <Icon name="circle-check-big" size={14} />
            </span>{" "}
            {t("download.completed", { name })}
          </p>
        )}
        {entry?.status === "unverified" && <p role="status">{t("download.unverified")}</p>}
        {entry?.status === "failed" && (
          <div className="error-card" role="alert" data-code={entry.error?.code}>
            <ErrorIcon tone="critical" />
            <strong>{t("download.failed", { name })}</strong>
            {errorTitle && <p>{errorTitle}</p>}
          </div>
        )}
        <p className="modal-actions">
          <button type="button" className="hairline" onClick={close}>
            {t("storage.dialog.cancel")}
          </button>
          {entry?.status === "unverified" ? (
            <button
              type="button"
              className="primary"
              onClick={() => void start(target.kind, target.catalogId, { check: true })}
            >
              {t("action.check_files")}
            </button>
          ) : (
            !busy &&
            entry?.status !== "completed" && (
              <button
                type="button"
                className="primary"
                disabled={!accepted}
                aria-disabled={!accepted}
                onClick={() =>
                  void start(target.kind, target.catalogId, {
                    confirm: entry?.status === "needs_confirm",
                  })
                }
              >
                <Icon name="download" size={14} /> {t("download.start")}
              </button>
            )
          )}
        </p>
      </div>
    </div>
  );
}
