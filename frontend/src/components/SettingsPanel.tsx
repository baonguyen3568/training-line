import { useEffect, useState } from "react";
import { api } from "../api";
import { appPrefix } from "../api/client";
import type { HealthInfo, LanInfo, SecretField, SessionInfo, SettingsView } from "../api/types";
import { getLocale, type Locale, t, tpOptional } from "../i18n";
import { switchLocale } from "../lib/localePref";
import { reloadForSignIn } from "../lib/session";
import { useWorkflow } from "../store/workflow";
import { ErrorIcon } from "./ErrorIcon";
import { Icon } from "./Icon";
import { toast } from "./QuickFixes";
import { errorText, SecretRow } from "./SecretRow";

const FIELDS: SecretField[] = ["HF_TOKEN", "NGC_API_KEY", "WANDB_API_KEY"];

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

const hhmm = (epochS: number) =>
  new Intl.DateTimeFormat(getLocale() === "vi" ? "vi-VN" : "en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: getLocale() !== "vi",
  }).format(new Date(epochS * 1000));

/**
 * D-8 "Phiên đăng nhập" (security): where this browser signs in from, sign out here or everywhere (the token
 * rotates; every browser opens the new link from the app's log), and, on the app's own machine only, the LAN
 * sign-in lock with "Mở khoá". After a sign-out the page reloads into the server's sign-in page.
 */
function SessionSection() {
  const [info, setInfo] = useState<SessionInfo | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true; // no state update after the panel is gone (a slow answer, a closed test)
    api.session().then(
      (i) => live && setInfo(i),
      () => live && setInfo(null),
    );
    return () => {
      live = false;
    };
  }, []);
  if (!info) return null;

  const act = async (fn: () => Promise<unknown>, after: () => void) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      after();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  const lock = info.lan_login;
  const locked = !!lock?.locked_until && lock.locked_until * 1000 > Date.now();

  return (
    <section aria-labelledby="set-session">
      <h2 id="set-session">
        <Icon name="shield-check" size={16} /> {t("session.section_title")}
      </h2>
      {/* the same two-column grid as Kết nối / Do quản trị viên đặt (design D-8 r1) */}
      <div className="set-row">
        <span className="set-label">{t("session.row_access")}</span>
        <div className="set-control">
          {info.origin_class === "local" ? t("session.origin_local") : t("session.origin_lan")}
        </div>
      </div>
      {((locked && lock?.locked_until) || note) && (
        <div className="set-row" data-testid={locked ? "lan-lock" : undefined}>
          <span className="set-label">{t("session.row_lan")}</span>
          <div className="set-control">
            {locked && lock?.locked_until ? (
              <>
                <p className="set-state">
                  <span className="si si-warn" aria-hidden="true">
                    <Icon name="triangle-alert" size={14} />
                  </span>{" "}
                  {t("session.lan_locked", {
                    time: hhmm(lock.locked_until),
                    failure_count: lock.recent_failures,
                  })}
                </p>
                <button
                  type="button"
                  className="hairline"
                  disabled={busy}
                  onClick={() =>
                    void act(api.unlockLan, () => {
                      setInfo({ ...info, lan_login: { locked_until: null, recent_failures: 0 } });
                      setNote(t("session.unlock_done"));
                    })
                  }
                >
                  {t("session.unlock_lan")}
                </button>
              </>
            ) : (
              <p className="set-state" role="status">
                <span className="si si-completed" aria-hidden="true">
                  <Icon name="circle-check-big" size={14} />
                </span>{" "}
                {note}
              </p>
            )}
          </div>
        </div>
      )}
      <div className="set-row">
        <span className="set-label">{t("session.row_sign_out")}</span>
        <div className="set-control">
          {confirmAll ? (
            <div role="alertdialog" aria-labelledby="revoke-all-q">
              <p id="revoke-all-q" className="set-state">
                {t("session.revoke_all_confirm")}
              </p>
              <p className="set-actions">
                <button type="button" className="hairline" onClick={() => setConfirmAll(false)}>
                  {t("storage.dialog.cancel")}
                </button>
                <button
                  type="button"
                  className="primary danger"
                  disabled={busy}
                  onClick={() => void act(api.revokeAllSessions, reloadForSignIn)}
                >
                  {t("session.revoke_all")}
                </button>
              </p>
            </div>
          ) : (
            <p className="set-actions">
              <button
                type="button"
                className="hairline"
                disabled={busy}
                onClick={() => void act(api.signOut, reloadForSignIn)}
              >
                {t("session.sign_out")}
              </button>
              <button type="button" className="hairline" onClick={() => setConfirmAll(true)}>
                {t("session.revoke_all")}
              </button>
            </p>
          )}
        </div>
      </div>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

/** Each language named in itself (design A5): the picker reads the same whichever language is on. */
const LANGUAGE_NAME: Record<Locale, string> = { vi: "Tiếng Việt", en: "English" };

/** Copy text; false when the browser gives no clipboard (http, permissions). */
async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * "Ngôn ngữ": choosing reloads the page in that language; an open workflow is kept as the draft the app offers to
 * restore, and if that cannot be saved the language stays and the user is told to save first.
 */
function LanguageRow() {
  const workflow = useWorkflow((st) => st.workflow);
  const [blocked, setBlocked] = useState(false);
  const current = getLocale();
  return (
    <div className="set-row">
      <span className="set-label" id="set-language-label">
        {t("settings.row_language")}
      </span>
      <div className="set-control">
        {/* native radios (keyboard arrows, one tab stop), drawn as the segmented control */}
        <fieldset className="seg seg-radio" aria-labelledby="set-language-label">
          {(Object.keys(LANGUAGE_NAME) as Locale[]).map((l) => (
            <label key={l} lang={l} className={l === current ? "on" : ""}>
              <input
                type="radio"
                name="app-language"
                className="sr-only"
                checked={l === current}
                onChange={() => setBlocked(!switchLocale(l, workflow))}
              />
              {LANGUAGE_NAME[l]}
            </label>
          ))}
        </fieldset>
        {blocked && (
          <p className="set-warning" role="alert">
            <span className="si si-warn" aria-hidden="true">
              <Icon name="triangle-alert" size={14} />
            </span>{" "}
            {t("settings.language_reload_confirm")}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * "Phiên bản": the version and commit this app runs (/api/health), and one button that copies exactly the bug
 * report's version + GPU fields, nothing else (design).
 */
function VersionRow() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [gpu, setGpu] = useState<string | undefined>(undefined);
  const [manual, setManual] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.health().then(
      (h) => live && setHealth(h),
      () => undefined,
    );
    api.bugReport(null, {}, getLocale()).then(
      (b) => live && setGpu(b.gpu),
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, []);
  if (!health) return null;
  const commit = health.commit
    ? `${health.commit.slice(0, 12)}${health.dirty ? "-dirty" : ""}`
    : "—";
  const text = [
    `${t("bug_report.field.app_version")}: ${health.app_version}`,
    ...(gpu ? [`${t("bug_report.field.gpu")}: ${gpu}`] : []),
  ].join("\n");
  const copy = async () => {
    if (await copyText(text)) {
      setManual(null);
      toast(t("settings.about.copied"));
    } else setManual(text); // shown to copy by hand
  };
  const [before, after = ""] = t("settings.about.version", {
    version: health.version,
    commit: "\u0000",
  }).split("\u0000");
  return (
    <div className="set-row">
      <span className="set-label">{t("settings.row_version")}</span>
      <div className="set-control">
        <p className="set-state">
          {before}
          <code>{commit}</code>
          {after}
        </p>
        <button type="button" className="hairline" onClick={() => void copy()}>
          <Icon name="copy" size={14} /> {t("settings.about.copy")}
        </button>
        {manual && <pre className="set-copy-manual">{manual}</pre>}
      </div>
    </div>
  );
}

/** "Ứng dụng" (design A5 r2): language and version together. */
function AppSection() {
  return (
    <section aria-labelledby="set-app">
      <h2 id="set-app">
        <Icon name="settings" size={16} /> {t("settings.section_app")}
      </h2>
      <LanguageRow />
      <VersionRow />
    </section>
  );
}

/** A small icon button that copies one value, with the copy itself as its name. */
function CopyButton({ value, label }: { value: string; label: string }) {
  return (
    <button
      type="button"
      className="help-btn inline"
      aria-label={label}
      title={label}
      onClick={() => void copyText(value).then((ok) => ok && toast(t("app.copied")))}
    >
      <Icon name="copy" size={14} />
    </button>
  );
}

/**
 * A5 "Truy cập từ máy khác": how other devices reach this app on the LAN and the CA they install, with its SHA-256
 * fingerprint to compare on that device (security). Only the app's own machine sees it (403 local_only elsewhere).
 */
function LanSection() {
  const [lan, setLan] = useState<LanInfo | null>(null);
  useEffect(() => {
    let live = true;
    api.lan().then(
      (l) => live && setLan(l),
      () => undefined, // 403 local_only (a LAN browser) or no endpoint: no section
    );
    return () => {
      live = false;
    };
  }, []);
  if (!lan) return null;
  return (
    <section aria-labelledby="set-lan">
      <h2 id="set-lan">
        <Icon name="globe" size={16} /> {t("settings.lan.section")}
      </h2>
      {!lan.enabled ? (
        <p className="muted">{t("settings.lan.off")}</p>
      ) : (
        <>
          <div className="set-row">
            <span className="set-label">{t("settings.lan.row_address")}</span>
            <div className="set-control">
              {lan.urls.map((u) => (
                <p key={u} className="set-state">
                  <code>{u}</code> <CopyButton value={u} label={t("settings.lan.copy_address")} />
                </p>
              ))}
            </div>
          </div>
          <div className="set-row" data-testid="lan-cert">
            <span className="set-label">{t("settings.lan.row_cert")}</span>
            <div className="set-control">
              {lan.ca ? (
                <>
                  <p>
                    <a className="btn ghost sm" href={`${appPrefix()}${lan.ca.download}`} download>
                      <Icon name="download" size={14} /> {t("settings.lan.ca_download")}
                    </a>
                  </p>
                  <p className="small muted">{t("settings.lan.fingerprint")}</p>
                  <p className="set-fingerprint">
                    <code>
                      {lan.ca.fingerprint_sha256.split(":").map((pair, i, all) => (
                        // a line may break only after a pair's colon, so a line never starts with ":"
                        // biome-ignore lint/suspicious/noArrayIndexKey: a fixed sequence of byte pairs
                        <span key={i}>
                          {pair}
                          {i < all.length - 1 && ":"}
                          <wbr />
                        </span>
                      ))}
                    </code>{" "}
                    <CopyButton
                      value={lan.ca.fingerprint_sha256}
                      label={t("settings.lan.copy_fingerprint")}
                    />
                  </p>
                  <p className="small muted">{t("settings.lan.howto")}</p>
                </>
              ) : (
                <p className="muted">{t("settings.lan.ca_missing")}</p>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}

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
      <SessionSection />
      <LanSection />
      <AppSection />
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
