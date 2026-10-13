import { describe, expect, it } from "vitest";
import { COMPACT_LANE, LANE_WIDTH, laneBoxes, occupiedLanes, toDisplayX, toModelX } from "./lanes";

describe("compact lanes (D4)", () => {
  it("empty lanes are 56px, occupied ones full width; an expanded lane is full while dragging", () => {
    const boxes = laneBoxes(new Set(["source", "train"]));
    expect(boxes.map((b) => b.width)).toEqual([260, 56, 260, 56, 56, 56, 56]);
    expect(laneBoxes(new Set(["source"]), "prepare")[1]?.width).toBe(LANE_WIDTH);
    expect(laneBoxes(new Set(), null, false).every((b) => b.width === LANE_WIDTH)).toBe(true);
  });

  it("display and saved x round-trip, and saved positions stay in the 260px lane space", () => {
    const boxes = laneBoxes(occupiedLanes([30, 2 * LANE_WIDTH + 30]));
    const trainX = 2 * LANE_WIDTH + 30;
    expect(toDisplayX(trainX, boxes)).toBe(LANE_WIDTH + COMPACT_LANE + 30);
    expect(toModelX(toDisplayX(trainX, boxes), boxes)).toBe(trainX);
    expect(toModelX(toDisplayX(30, boxes), boxes)).toBe(30);
  });
});
