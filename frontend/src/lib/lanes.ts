import type { Category } from "../api/types";

// Canvas lanes ①…⑦ (P010 §3). NodeSpec.category → lane id used by i18n `canvas.lane.<id>`.
export const LANES = ["source", "prepare", "train", "try", "quantize", "eval", "export"] as const;
export type Lane = (typeof LANES)[number];

const byCategory: Record<Category, Lane> = {
  source: "source",
  data: "prepare",
  train: "train",
  test: "try",
  quantize: "quantize",
  eval: "eval",
  export: "export",
  util: "export",
};

export const laneOf = (category: Category): Lane => byCategory[category];

export const LANE_WIDTH = 260;
export const ROW_HEIGHT = 170;

/** Lane-based layout: x from the lane, y from the order within the lane (no dagre for the common case). */
export function laneLayout(
  items: { id: string; category: Category }[],
): Map<string, { x: number; y: number }> {
  const rows = new Map<Lane, number>();
  const out = new Map<string, { x: number; y: number }>();
  for (const item of items) {
    const lane = laneOf(item.category);
    const row = rows.get(lane) ?? 0;
    rows.set(lane, row + 1);
    out.set(item.id, { x: LANES.indexOf(lane) * LANE_WIDTH + 30, y: 70 + row * ROW_HEIGHT });
  }
  return out;
}
