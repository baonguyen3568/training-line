import { expect, it } from "vitest";
import { formatDateTime, formatDuration } from "./format";

it("formats date-time per locale (vi: dd/mm/yyyy HH:MM, 24h)", () => {
  const ts = new Date(2026, 9, 12, 17, 35).getTime();
  expect(formatDateTime(ts, "vi")).toBe("12/10/2026 17:35");
  expect(formatDateTime(ts, "en")).toMatch(/10\/12\/2026.*5:35/);
});

it("formats durations", () => {
  expect(formatDuration(65_000)).toBe("1:05");
  expect(formatDuration(3_725_000)).toBe("1:02:05");
});
