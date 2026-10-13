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

/**
 * D4 compact lanes (design M2-6 r1 B3): a lane with no node shrinks to COMPACT_LANE on screen. Display-only —
 * saved positions stay in the canonical 260px-lane space, so files and the classic UI are unchanged.
 */
export const COMPACT_LANE = 56;

export interface LaneBox {
  lane: Lane;
  x: number;
  width: number;
  compact: boolean;
}

/** Canonical lane index of a saved x position. */
export function laneIndexAt(x: number): number {
  return Math.min(LANES.length - 1, Math.max(0, Math.floor(x / LANE_WIDTH)));
}

/** Screen geometry of the 7 lanes: occupied (or currently expanded) lanes full width, the others compact. */
export function laneBoxes(
  occupied: ReadonlySet<Lane>,
  expand?: Lane | null,
  compact = true,
): LaneBox[] {
  let x = 0;
  return LANES.map((lane) => {
    const small = compact && !occupied.has(lane) && lane !== expand;
    const box = { lane, x, width: small ? COMPACT_LANE : LANE_WIDTH, compact: small };
    x += box.width;
    return box;
  });
}

/** Saved x → screen x. */
export function toDisplayX(x: number, boxes: readonly LaneBox[]): number {
  const i = laneIndexAt(x);
  const box = boxes[i];
  if (!box) return x;
  const offset = x - i * LANE_WIDTH;
  return box.x + (box.compact ? (offset * box.width) / LANE_WIDTH : offset);
}

/** Screen x → saved x (inverse of toDisplayX; positions left of the canvas map into lane ①). */
export function toModelX(x: number, boxes: readonly LaneBox[]): number {
  const i = Math.max(
    0,
    boxes.findLastIndex((b) => x >= b.x),
  );
  const box = boxes[i];
  if (!box) return x;
  const offset = x - box.x;
  return i * LANE_WIDTH + (box.compact ? (offset * LANE_WIDTH) / box.width : offset);
}

/** Lanes that hold at least one node (by saved position, so a node dragged into a lane opens it). */
export function occupiedLanes(xs: Iterable<number>): Set<Lane> {
  const out = new Set<Lane>();
  for (const x of xs) {
    const lane = LANES[laneIndexAt(x)];
    if (lane) out.add(lane);
  }
  return out;
}
