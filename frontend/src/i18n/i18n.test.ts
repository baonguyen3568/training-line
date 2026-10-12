import { describe, expect, it } from "vitest";
import { setLocale, t, tp } from ".";

describe("i18n", () => {
  it("defaults to Vietnamese and interpolates", () => {
    setLocale("vi");
    expect(t("header.ram", { used_gb: 79, total_gb: 121 })).toBe("RAM 79/121 GB");
  });
  it("treats dotted node type ids as one key", () => {
    expect(tp(["node", "train.lora_vlm", "title"])).toBe("Fine-tune nhẹ (LoRA)");
  });
  it("returns the key when missing", () => {
    expect(t("does.not.exist")).toBe("does.not.exist");
  });
  it("switches to English", () => {
    setLocale("en");
    expect(tp(["node", "train.lora_vlm", "title"])).not.toBe("Fine-tune nhẹ (LoRA)");
    setLocale("vi");
  });
});
