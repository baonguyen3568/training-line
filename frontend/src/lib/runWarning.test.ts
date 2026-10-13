import { describe, expect, it } from "vitest";
import { runWarningText } from "./runWarning";

describe("runWarningText", () => {
  it("renders the full explanation when every param is known, with the node title and a local date", () => {
    const r = runWarningText(
      { code: "CACHE-001", params: { node: "n1", saved_at: 1791795518, est_minutes: 5 } },
      () => "Chọn model",
    );
    expect(r.title).toBe("Kết quả cũ bị hỏng, sẽ chạy lại");
    expect(r.what).toContain('"Chọn model"');
    expect(r.what).toContain("5 phút");
    expect(r.what).not.toContain("1791795518");
  });

  it("never shows null: drops the explanation when a param is missing and no variant exists", () => {
    const r = runWarningText(
      { code: "CACHE-001", params: { node: "n1", saved_at: 1, est_minutes: null } },
      () => "X",
    );
    expect(r.title).toBe("Kết quả cũ bị hỏng, sẽ chạy lại");
    expect(r.what).toContain("chạy lại bước này."); // what_no_estimate variant
    expect(r.what ?? "").not.toMatch(/null|\{est_minutes\}/);
  });
});
