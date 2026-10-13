import { describe, expect, it } from "vitest";
import type { ValidationResult } from "../api/types";
import {
  durationRunCount,
  estimateLine,
  formatMinutesRange,
  formatShortDeadline,
  workflowDurationText,
} from "./format";

type Est = ValidationResult["estimate"];
const base: Est = {
  ram_gb_peak: null,
  disk_gb: null,
  duration_min: null,
  gpu_exclusive: true,
  source: "unknown",
};

describe("measured duration", () => {
  it("reads hours and minutes, not decimal hours", () => {
    expect(formatMinutesRange({ lo: 97, hi: 112 })).toBe("~1 giờ 37 phút – 1 giờ 52 phút");
    expect(formatMinutesRange({ lo: 45, hi: 120 })).toBe("~45 phút – 2 giờ");
    expect(formatMinutesRange({ lo: 20.4, hi: 35.6 })).toBe("~20–36 phút");
    expect(formatMinutesRange(null)).toBe("chưa có số đo");
  });

  it("a narrow range reads as one number (< 5 min or < 5 % of hi)", () => {
    expect(formatMinutesRange({ lo: 114, hi: 115 })).toBe("~1 giờ 55 phút");
    expect(formatMinutesRange({ lo: 38.5, hi: 39.4 })).toBe("~39 phút");
    expect(formatMinutesRange({ lo: 189, hi: 191 })).toBe("~3 giờ 11 phút");
    expect(formatMinutesRange({ lo: 32, hi: 37 })).toBe("~32–37 phút");
  });

  it("shows measured time even when RAM/disk source is unknown", () => {
    const est: Est = {
      ...base,
      duration: {
        duration_min: { lo: 97, hi: 112 },
        per_node: {
          n4: { steps: 300, minutes: { lo: 97, hi: 112 }, based_on: ["run-a", "run-b"] },
        },
        unmeasured: ["n6"],
      },
    };
    expect(workflowDurationText(est)).toBe("~1 giờ 37 phút – 1 giờ 52 phút");
    expect(durationRunCount(est)).toBe(2);
  });

  it("a missing train measurement is unknown, never a partial sum", () => {
    expect(workflowDurationText({ ...base, duration: null })).toBe("chưa có số đo");
    expect(
      workflowDurationText({
        ...base,
        source: "measured",
        duration_min: { lo: 1, hi: 2 },
        duration: { duration_min: null, per_node: {}, unmeasured: [] },
      }),
    ).toBe("chưa có số đo");
  });

  it("older servers without `duration` keep the source gate", () => {
    expect(workflowDurationText({ ...base, duration_min: { lo: 10, hi: 20 } })).toBe(
      "chưa có số đo",
    );
    expect(
      workflowDurationText({ ...base, source: "formula", duration_min: { lo: 10, hi: 20 } }),
    ).toBe("~10–20 phút");
  });

  it("estimate line lists known numbers and names the missing ones once", () => {
    expect(estimateLine(base)).toBe("chưa có số đo thời gian, RAM, ổ đĩa");
    expect(
      estimateLine({
        ...base,
        source: "measured",
        ram_gb_peak: { lo: 40, hi: 48 },
        disk_gb: { lo: 30, hi: 35 },
        duration: null,
      }),
    ).toBe("Dự kiến: RAM ~40–48 GB · Ổ đĩa ~30–35 GB · Chưa có số đo: thời gian");
  });
});

describe("short deadline", () => {
  it("reads like design's examples, and leaves unknown shapes alone", () => {
    expect(formatShortDeadline("2026-10-27 09:00", "vi")).toBe("09:00 27/10");
    expect(formatShortDeadline("2026-10-27 09:00", "en")).toBe("Oct 27, 09:00");
    expect(formatShortDeadline("soon", "vi")).toBe("soon");
  });
});
