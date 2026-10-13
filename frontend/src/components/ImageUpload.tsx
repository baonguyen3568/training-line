import { useState } from "react";
import { api } from "../api";
import { ApiError, uploadUrl } from "../api/client";
import type { RenderedError } from "../api/types";
import { getLocale, t } from "../i18n";
import { untrustedText } from "../lib/safeText";

const MAX_BYTES = 20 * 1024 * 1024; // the server enforces it too (413); checked here only to answer early
const ACCEPT = "image/png,image/jpeg,image/webp";

/**
 * "Chạy thử" image param (B-7): the file goes to POST /api/uploads (raw body); the param keeps only the
 * server-made upload_id, never a path or the file name. The preview is the server's re-encoded PNG.
 */
export function ImageUpload({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (uploadId: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ title: string; what?: string } | null>(null);
  const [name, setName] = useState<string | null>(null);

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    if (file.size > MAX_BYTES) {
      setError({ title: t("trial.image_too_large") });
      return;
    }
    setBusy(true);
    try {
      const info = await api.upload(file, getLocale());
      setName(untrustedText(info.label, 60));
      onChange(info.upload_id);
    } catch (e) {
      const err =
        e instanceof ApiError ? (e.body as { error?: RenderedError } | null)?.error : undefined;
      setError(
        err?.title ? { title: err.title, what: err.what } : { title: t("trial.upload_failed") },
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="field image-upload">
      <label htmlFor={id}>{label}</label>
      {value && (
        <figure className="upload-preview">
          <img src={uploadUrl(value)} alt={name ?? t("trial.image_alt")} />
          <figcaption>
            {name ?? t("trial.image_selected")}{" "}
            <button type="button" className="link" onClick={() => onChange("")}>
              {t("trial.image_remove")}
            </button>
          </figcaption>
        </figure>
      )}
      <input
        id={id}
        type="file"
        accept={ACCEPT}
        disabled={busy}
        onChange={(e) => {
          void pick(e.target.files?.[0]);
          e.target.value = ""; // picking the same file again still uploads
        }}
      />
      {busy && <p className="muted small">{t("trial.uploading")}</p>}
      {error && (
        <p className="error-inline" role="alert">
          <strong>{error.title}</strong>
          {error.what && <> {error.what}</>}
        </p>
      )}
    </div>
  );
}
