import { useId } from "react";
import { t } from "../i18n";
import { Icon } from "./Icon";

/**
 * "Lần chạy thử" (ADR-021 quick trial, design r2): a run of a few steps to check the model and data run at all, never
 * a training result. A static tag, never button-like (no icon, no border, no hover: "Chạy thử nhanh" is the play
 * button elsewhere); its "?" says what a trial run is.
 */
export function TrialChip() {
  const pop = `trial-${useId().replace(/:/g, "")}`;
  return (
    <span className="trial-chip-wrap">
      <span className="trial-chip">{t("run.trial_chip")}</span>
      <button
        type="button"
        className="help-btn inline"
        popoverTarget={pop}
        aria-label={t("d4.explain")}
      >
        <Icon name="circle-question-mark" size={14} />
      </button>
      <span id={pop} popover="auto" className="pop">
        <strong>{t("run.trial_chip")}</strong>
        <span className="pop-p">{t("run.trial_help")}</span>
      </span>
    </span>
  );
}
