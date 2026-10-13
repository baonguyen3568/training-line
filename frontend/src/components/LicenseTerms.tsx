import type { LicenseObligation } from "../api/types";
import { tpOptional } from "../i18n";
import { licenseName } from "../lib/licenses";
import { Icon } from "./Icon";

// the verdict's icon carries the status ink; its words stay in text ink (Ink-Not-Label)
const BADGE = {
  commercial_ok: { tone: "si-completed", icon: "circle-check-big" },
  conditional: { tone: "si-warn", icon: "triangle-alert" },
  non_commercial: { tone: "si-bad", icon: "ban" },
  unknown: { tone: "si-muted", icon: "badge-question-mark" },
} as const;

/** Params a duty's copy needs: the verbatim notice, the parts under another licence ("tokenizer"). */
const dutyText = (o: LicenseObligation) =>
  tpOptional(["license", "obligation", o.id], {
    notice: o.text ?? "",
    component: (o.parts ?? []).join(", "),
  });

/**
 * A licence before "I agree" (design G-5 #4): its name, the `license.badge.*` verdict and the duties people have
 * to keep, so the checkbox has something to read. Duties without approved copy are left out, never shown as ids.
 */
export function LicenseTerms({
  license,
  licenseClass,
  obligations = [],
}: {
  license: string;
  licenseClass?: keyof typeof BADGE | undefined;
  obligations?: LicenseObligation[] | undefined;
}) {
  const withCopy = obligations.flatMap((o) => {
    const text = dutyText(o);
    return text ? [{ o, text }] : [];
  });
  // design 4fd5f71: what to keep while using it, then what to keep when sharing (the server already dropped
  // `cosmos` duties for models outside that family; the ones left are about sharing)
  const groups = [
    {
      key: "obligations_use",
      duties: withCopy.filter(({ o }) => (o.when ?? "always") === "always"),
    },
    {
      key: "obligations_distribute",
      duties: withCopy.filter(({ o }) => o.when === "distribute" || o.when === "cosmos"),
    },
  ].filter((g) => g.duties.length > 0);
  const badge = licenseClass ? tpOptional(["license", "badge", licenseClass]) : undefined;
  return (
    <div className="license-terms">
      <p>
        <strong>{licenseName(license)}</strong>
        {badge && licenseClass && (
          <span className="badge license-verdict">
            <span className={`si ${BADGE[licenseClass].tone}`} aria-hidden="true">
              <Icon name={BADGE[licenseClass].icon} size={12} />
            </span>{" "}
            {badge}
          </span>
        )}
      </p>
      {groups.length > 0 && <p className="small">{tpOptional(["license", "obligations_title"])}</p>}
      {groups.map((g) => (
        <div key={g.key}>
          <p className="small muted">{tpOptional(["license", g.key])}</p>
          <ul className="small license-duties">
            {g.duties.map(({ o, text }) => (
              <li key={`${o.licence}-${o.id}-${(o.parts ?? []).join()}`}>
                {text}
                {o.id === "prohibited_use_policy_link" && o.url?.startsWith("https://") && (
                  <>
                    {" "}
                    <a href={o.url} target="_blank" rel="noopener noreferrer">
                      {tpOptional(["license", "policy_link"])}
                    </a>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
