import { tpOptional } from "../i18n";
import { untrustedText } from "./safeText";

// Licence ids → display names: design's copy license.name.<id> (security: names live in i18n), else these labels
// (until that copy exists), else the id as written, never invented. Names only: the class and obligations always
// come from the server (security's obligations table), never from here.
const LICENSE_NAME: Record<string, string> = {
  "apache-2.0": "Apache-2.0",
  mit: "MIT",
  "cc-by-4.0": "CC-BY-4.0",
  "nvidia-open-model-license": "NVIDIA Open Model License",
  "openmdw-1.1": "OpenMDW-1.1",
};
export const licenseName = (id: string) =>
  tpOptional(["license", "name", id]) ?? LICENSE_NAME[id] ?? id;

/** "Tác giả ghi: “…”": a user-added asset's licence in its author's words (untrusted: plain text, clamped), or
 * undefined while the copy is missing (the line hides, never a key). */
export const cardSays = (text: string): string | undefined =>
  tpOptional(["license", "card_says"], { text: untrustedText(text, 200) });
/** The whole of it, for the line's title (still visible-escaped; bounded so a huge card cannot flood it). */
export const cardFull = (text: string): string => untrustedText(text, 2000);
