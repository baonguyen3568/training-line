import { Icon } from "./Icon";

type Tone = "critical" | "warn" | "info";
const CLASS: Record<Tone, string> = { critical: "si-bad", warn: "si-warn", info: "si-info" };

/** Leading glyph of an .error-card (design D7 r1: hairline card + glyph in status ink, no side stripe). */
export function ErrorIcon({ tone = "critical" }: { tone?: Tone }) {
  return (
    <span className={`ec-icon si ${CLASS[tone]}`} aria-hidden="true">
      <Icon name={tone === "info" ? "info" : "triangle-alert"} size={16} />
    </span>
  );
}
