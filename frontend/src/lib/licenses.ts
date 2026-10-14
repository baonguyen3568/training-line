import { tpOptional } from "../i18n";
import { untrustedText } from "./safeText";

// Licence ids → display names: design's copy license.name.<id> (security: names live in i18n), else the id as
// written, never invented. Names only: the class and obligations always come from the server (security's
// obligations table). The id is one path segment: ids with dots ("llama3.1", "lfm1.0") are never split.
export const licenseName = (id: string): string => tpOptional(["license", "name", id]) ?? id;

/** "Tác giả ghi: “…”": a user-added asset's licence in its author's words (untrusted: plain text, clamped), or
 * undefined while the copy is missing (the line hides, never a key). */
export const cardSays = (text: string): string | undefined =>
  tpOptional(["license", "card_says"], { text: untrustedText(text, 200) });
/** The whole of it, for the line's title (still visible-escaped; bounded so a huge card cannot flood it). */
export const cardFull = (text: string): string => untrustedText(text, 2000);
