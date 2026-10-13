import { useEffect, useState } from "react";
import { api } from "../api";
import type { SecretState } from "../api/types";
import { t } from "../i18n";
import { Icon } from "./Icon";
import { SecretRow } from "./SettingsPanel";

const HF = "https://huggingface.co";

/** An external page in a new tab; the app never sees what the user types there. */
function Open({ href, label }: { href: string; label: string }) {
  return (
    <a className="btn ghost sm" href={href} target="_blank" rel="noopener noreferrer">
      <Icon name="external-link" size={14} /> {label}
    </a>
  );
}

/**
 * The gated model flow (design G-5 r3 #2, copy `gated.*`): Hugging Face account, the model's terms, a read-only
 * token, then that token pasted into the app (the same write-only row as Settings). The server cannot yet tell
 * whether the account has access, so the user says when they are done and downloads; a refusal comes back as
 * NET-001 on the dialog.
 */
export function GatedSteps({
  repo,
  model,
  publisher,
}: {
  repo: string;
  model: string;
  publisher: string;
}) {
  const [hf, setHf] = useState<SecretState | null>(null);
  useEffect(() => {
    let live = true;
    api.settings().then(
      (v) => live && setHf(v.secrets.HF_TOKEN),
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, []);
  const step = (n: number) => t("gated.step_count", { step: n, total_count: 4 });

  return (
    <details className="gated-steps">
      {/* a native summary already tells assistive tech whether it is open */}
      <summary className="btn ghost sm">
        <span className="chev" aria-hidden="true">
          <Icon name="chevron-right" size={14} />
        </span>{" "}
        {t("gated.title", { model })}
      </summary>
      <ol>
        <li>
          <p className="small muted">{step(1)}</p>
          <strong>{t("gated.step1.title")}</strong>
          <p className="small">{t("gated.step1.body")}</p>
          <Open href={`${HF}/join`} label={t("gated.step1.open")} />
        </li>
        <li>
          <p className="small muted">{step(2)}</p>
          <strong>{t("gated.step2.title")}</strong>
          <p className="small">{t("gated.step2.body")}</p>
          <p className="small muted">{t("gated.step2.shares_info", { publisher })}</p>
          <Open href={`${HF}/${repo}`} label={t("gated.step2.open")} />
        </li>
        <li>
          <p className="small muted">{step(3)}</p>
          <strong>{t("gated.step3.title")}</strong>
          <p className="small">{t("gated.step3.body")}</p>
          <p className="small">{t("gated.step3.never_write")}</p>
          <p className="small muted">{t("gated.step3.risk")}</p>
          <Open href={`${HF}/settings/tokens`} label={t("gated.step3.open")} />
        </li>
        <li>
          <p className="small muted">{step(4)}</p>
          <strong>{t("gated.step4.title")}</strong>
          <p className="small">{t("gated.step4.body")}</p>
          {hf && <SecretRow field="HF_TOKEN" state={hf} onChange={setHf} />}
        </li>
      </ol>
    </details>
  );
}
