import { useId } from "react";
import { t } from "../i18n";
import { Icon } from "./Icon";

/**
 * "Chạy thử" (ADR-021 quick trial, design): a run of a few steps to check the model and data run at all, never a
 * training result. Text ink, hairline, no accent; its "?" says what a trial run is.
 */
export function TrialChip() {
  const pop = `trial-${useId().replace(/:/g, "")}`;
  return (
    <span className="trial-chip-wrap">
      <span className="chip trial-chip">
        <Icon name="play" size={12} /> {t("run.trial_chip")}
      </span>
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
