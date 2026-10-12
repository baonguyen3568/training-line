import { describe, expect, it } from "vitest";
import { identifierView, stripInvisible, untrustedText } from "./safeText";

describe("safeText", () => {
  it("escapes bidi overrides and zero-width chars (Trojan Source)", () => {
    expect(stripInvisible("abc‮def​g")).toBe("abc\\u202Edef\\u200Bg");
  });

  it("keeps Vietnamese text intact", () => {
    expect(untrustedText("Fine-tune nhẹ cho hoá đơn")).toBe("Fine-tune nhẹ cho hoá đơn");
  });

  it("truncates by code points", () => {
    expect(untrustedText("x".repeat(500))).toBe(`${"x".repeat(60)}…`);
  });

  it("flags non-ASCII identifiers such as Cyrillic homoglyphs", () => {
    const v = identifierView("lorа-trainer"); // Cyrillic "а"
    expect(v.suspicious).toBe(true);
    expect(v.text).toBe("lor\\u0430-trainer");
    expect(identifierView("acme.ocr_clean").suspicious).toBe(false);
  });
});
