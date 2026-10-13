import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { t } from "../i18n";
import { licenseName } from "../lib/licenses";
import { missingDownloads } from "../lib/requestRun";
import { downloadKey, downloadPercent, useDownloads } from "../store/downloads";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import type { DownloadTarget } from "./DownloadDialog";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";

const BUSY = ["starting", "queued", "downloading", "verifying"];

/**
 * G-5 "Bấm Chạy khi thiếu" (P018 §3.3, design 9015477): what the workflow still needs on this machine, with sizes
 * and licences, one "I agree" for all of them, then one download after another; the run starts only when every one
 * completed, and a failure stops the chain without running. A gated model is never downloaded in the chain (the
 * user agrees to its terms on Hugging Face first): its row opens its own download, and "Tải rồi chạy" waits for it.
 */
export function RunMissingDialog({ onClose }: { onClose: () => void }) {
  // the list as it was when Run was pressed: rows stay while each completed one leaves validate's warns
  const [targets] = useState<DownloadTarget[]>(() => missingDownloads());
  const entries = useDownloads((s) => s.entries);
  const [accepted, setAccepted] = useState(false);
  const [chaining, setChaining] = useState(false);
  const [stoppedAt, setStoppedAt] = useState<string | null>(null);
  const opener = useRef<Element | null>(document.activeElement);

  const close = useCallback(() => {
    onClose();
    (opener.current as HTMLElement | null)?.focus?.();
  }, [onClose]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !chaining) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close, chaining]);

  const entryOf = (x: DownloadTarget) => entries[downloadKey(x.kind, x.catalogId)];
  const done = (x: DownloadTarget) => entryOf(x)?.status === "completed";
  const gatedPending = targets.filter((x) => x.gated && !done(x));
  const chain = targets.filter((x) => !x.gated && !done(x));
  const totalGb = targets
    .filter((x) => !done(x) && !x.prepare)
    .reduce((sum, x) => sum + (x.sizeGb ?? 0), 0);
  const licenses = useMemo(
    () => [...new Set(targets.map((x) => x.license).filter((l): l is string => !!l))],
    [targets],
  );
  const stopped = targets.find((x) => downloadKey(x.kind, x.catalogId) === stoppedAt);

  const go = async () => {
    setChaining(true);
    setStoppedAt(null);
    const failed = await useDownloads.getState().downloadAll(chain);
    setChaining(false);
    if (failed) {
      setStoppedAt(failed);
      return;
    }
    await useWorkflow.getState().validate();
    onClose();
    void useWorkflow.getState().startRun();
  };

  return (
    <div className="modal-scrim">
      <div
        className="modal download run-missing"
        role="dialog"
        aria-modal="true"
        aria-labelledby="run-missing-title"
      >
        <h2 id="run-missing-title">
          <Icon name="download" size={16} /> {t("download.missing_title")}
        </h2>
        <p>{t("download.missing_intro", { item_count: targets.length })}</p>
        <ul className="missing-list">
          {targets.map((x) => {
            const e = entryOf(x);
            const percent = downloadPercent(e);
            return (
              <li key={downloadKey(x.kind, x.catalogId)} data-status={e?.status ?? "idle"}>
                <span className="missing-name">
                  <strong>{e?.name ?? x.name}</strong>
                  {x.prepare ? (
                    <span className="muted small"> · {t("download.prepare_only")}</span>
                  ) : (
                    x.sizeGb !== undefined && (
                      <span className="muted small"> · {x.sizeGb.toFixed(1)} GB</span>
                    )
                  )}
                  {x.license && (
                    <span className="muted small"> · {licenseName(e?.license ?? x.license)}</span>
                  )}
                </span>
                {e?.status === "completed" ? (
                  <span className="small" role="status">
                    <span className="si si-completed" aria-hidden="true">
                      <Icon name="circle-check-big" size={14} />
                    </span>{" "}
                    {t("download.completed", { name: e.name ?? x.name })}
                  </span>
                ) : e?.status === "verifying" ? (
                  <span className="small">{t("download.verifying")}</span>
                ) : e && BUSY.includes(e.status) ? (
                  <span className="download-progress">
                    <progress max={100} value={percent ?? 0} aria-label={x.name} />
                    <span className="small">
                      {t("download.progress", {
                        done_gb: ((e.bytesDone ?? 0) / 1e9).toFixed(1),
                        total_gb: ((e.bytesTotal ?? 0) / 1e9).toFixed(1),
                        file_done_count: e.filesDone ?? 0,
                        file_count: e.filesTotal ?? "?",
                      })}
                    </span>
                  </span>
                ) : e?.status === "failed" ? (
                  <span className="small">
                    <ErrorIcon tone="critical" /> {t("download.failed", { name: x.name })}
                  </span>
                ) : (
                  x.gated && (
                    <span className="small">
                      {t("download.missing_gated", { name: x.name })}{" "}
                      <button
                        type="button"
                        className="link"
                        onClick={() => {
                          // its own dialog: the gated intro, licence and the user's own HF sign-in
                          onClose();
                          useUi.getState().setDownloadTarget(x);
                        }}
                      >
                        {t("action.download")}
                      </button>
                    </span>
                  )
                )}
              </li>
            );
          })}
        </ul>
        {totalGb > 0 && (
          <p className="small">{t("download.missing_total", { size_gb: totalGb.toFixed(1) })}</p>
        )}
        {licenses.length > 0 && (
          <p>
            {t("download.license_lead")}{" "}
            {licenses.map((l, i) => (
              <span key={l}>
                {i > 0 && ", "}
                <strong>{licenseName(l)}</strong>
              </span>
            ))}
          </p>
        )}
        {!chaining && (
          <label className="field check">
            <input
              type="checkbox"
              checked={accepted}
              onChange={(e) => setAccepted(e.target.checked)}
            />
            {t("download.license_accept_all")}
          </label>
        )}
        {stopped && (
          <div className="error-card" role="alert">
            <ErrorIcon tone="critical" />
            <strong>{t("download.chain_stopped", { name: stopped.name })}</strong>
          </div>
        )}
        <p className="modal-actions">
          <button type="button" className="hairline" onClick={close} disabled={chaining}>
            {t("storage.dialog.cancel")}
          </button>
          {!chaining && (
            <button
              type="button"
              className="primary"
              disabled={!accepted || gatedPending.length > 0}
              aria-disabled={!accepted || gatedPending.length > 0}
              onClick={() => void go()}
            >
              <Icon name="download" size={14} /> {t("download.download_and_run")}
            </button>
          )}
        </p>
      </div>
    </div>
  );
}
