// Helpers for rendering untrusted strings (workflow files, third-party node manifests).
// React already escapes text; these guard against what escaping does not cover:
// bidi/control characters ("Trojan Source") and homoglyph identifiers (P003 D16, 009 annex F6d).

// C0/C1 controls (except \t \n), zero-width + LRM/RLM, bidi embeddings/overrides, bidi isolates, BOM.
// biome-ignore lint/suspicious/noControlCharactersInRegex: matching control characters is the purpose.
const INVISIBLE = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F​-‏‪-‮⁦-⁩﻿]/g;

const hex = (ch: string) =>
  `\\u${(ch.codePointAt(0) ?? 0).toString(16).toUpperCase().padStart(4, "0")}`;

/** Replace invisible/bidi characters with a visible escape so the text cannot reorder itself. */
export function stripInvisible(text: string): string {
  return text.replace(INVISIBLE, hex);
}

/** Display text from an untrusted source: visible escapes, truncated by code points. */
export function untrustedText(text: string, maxChars = 60): string {
  const visible = Array.from(stripInvisible(text));
  return visible.length > maxChars ? `${visible.slice(0, maxChars).join("")}…` : visible.join("");
}

export interface IdentifierView {
  text: string;
  /** True when the identifier contains anything outside printable ASCII (possible homoglyph). */
  suspicious: boolean;
}

/**
 * Identifiers (node_type, publisher/pack) must be ASCII per D16 (`^[a-z0-9][a-z0-9_.-]{0,63}$`, enforced by
 * the server). This is the second line of defence: any non-ASCII code point is shown escaped and flagged.
 */
export function identifierView(id: string, maxChars = 60): IdentifierView {
  let suspicious = false;
  const out = Array.from(id)
    .map((ch) => {
      const cp = ch.codePointAt(0) ?? 0;
      if (cp >= 0x20 && cp <= 0x7e) return ch;
      suspicious = true;
      return hex(ch);
    })
    .join("");
  const chars = Array.from(out);
  const text = chars.length > maxChars ? `${chars.slice(0, maxChars).join("")}…` : out;
  return { text, suspicious };
}
