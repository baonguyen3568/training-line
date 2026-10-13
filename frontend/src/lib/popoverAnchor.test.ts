import { describe, expect, it } from "vitest";
import { placePopover } from "./popoverAnchor";

const rect = (x: number, y: number, w: number, h: number) =>
  ({
    left: x,
    top: y,
    right: x + w,
    bottom: y + h,
    width: w,
    height: h,
    x,
    y,
    toJSON() {},
  }) as DOMRect;

describe("placePopover", () => {
  it("opens within 8px below its own trigger, not centred on the page", () => {
    const trigger = document.createElement("button");
    trigger.getBoundingClientRect = () => rect(100, 200, 20, 20);
    const pop = document.createElement("div");
    Object.defineProperty(pop, "offsetWidth", { value: 300 });
    Object.defineProperty(pop, "offsetHeight", { value: 120 });
    Object.defineProperty(window, "innerWidth", { value: 1440, configurable: true });
    Object.defineProperty(window, "innerHeight", { value: 900, configurable: true });
    placePopover(pop, trigger);
    expect(Number.parseFloat(pop.style.top) - 220).toBeLessThanOrEqual(8);
    expect(Math.abs(Number.parseFloat(pop.style.left) - 100)).toBeLessThanOrEqual(20);
    expect(pop.dataset.side).toBe("below");
  });

  it("flips above near the bottom edge and stays inside the right edge", () => {
    const trigger = document.createElement("button");
    trigger.getBoundingClientRect = () => rect(1420, 860, 16, 16);
    const pop = document.createElement("div");
    Object.defineProperty(pop, "offsetWidth", { value: 300 });
    Object.defineProperty(pop, "offsetHeight", { value: 120 });
    placePopover(pop, trigger);
    expect(pop.dataset.side).toBe("above");
    expect(Number.parseFloat(pop.style.top) + 120).toBeLessThanOrEqual(860);
    expect(Number.parseFloat(pop.style.left) + 300).toBeLessThanOrEqual(1440 - 8);
  });
});
