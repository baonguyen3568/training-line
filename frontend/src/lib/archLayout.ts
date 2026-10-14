/**
 * Geometry of the training-architecture diagram (design spec I §2), with no renderer: the app draws it with React
 * (components/ArchDiagram.tsx), the site with Astro (site/src/components/ArchDiagram.astro), from this one layout so
 * the two never drift. Copy comes in through `tr` (the app's i18n or the site's exported copy).
 */

/** `train_arch` of configs/reference_pipelines.yaml v2 (common/reference_pipelines.py, P023 §5). */
export type BlockState = "frozen" | "lora" | "full" | "full_low_lr" | "n/a";
export type Quant = "nf4" | "fp8" | "nvfp4";
export type Extra = "teacher" | "reference" | "reference_shared" | "rollout" | "reward";
export interface ArchStage {
  name_key?: string;
  vision: BlockState;
  projector: BlockState;
  lm: BlockState;
  quant?: Quant | null;
  fake_quant?: boolean;
}
export interface TrainArch extends ArchStage {
  extras?: Extra[];
  stages?: ArchStage[];
}

/** `key` is a dotted copy key ("arch.block.vision"); `{name}` placeholders are filled from `params`. */
export type Translate = (key: string, params?: Record<string, string | number>) => string;

export type ArchShape =
  | { kind: "rect"; x: number; y: number; w: number; h: number; rx: number; cls?: string }
  | {
      kind: "text";
      x: number;
      y: number;
      lines: string[];
      dy: number;
      anchor?: "middle" | "end";
      cls?: string;
    }
  | { kind: "icon"; name: string; x: number; y: number; size: number; cls?: string }
  | { kind: "wire"; x1: number; y1: number; x2: number; y2: number };

export interface ArchLayout {
  width: number;
  height: number;
  /** The diagram's accessible name, built from the data (spec I §2 a11y). */
  label: string;
  groups: { cls: string; shapes: ArchShape[] }[];
}

const BLOCKS = ["vision", "projector", "lm"] as const;

// geometry (viewBox units = px at 1:1): blocks 64×34 (spec minimum width) keep 11px text in two lines, the gaps
// leave room for a LoRA pill past a block's right edge; 3 blocks + a stage label fit a 280px card
const BW = 64;
const BH = 34;
const GAP = 22;
const LABEL_W = 24;
const ROW_H = 60;
const SIDE_H = 30;
const PAD_TOP = 12;

const stateKey = (s: BlockState) => (s === "n/a" ? "n_a" : s);
const blockCls = (s: BlockState) =>
  s === "full" ? "train" : s === "full_low_lr" ? "train-low" : s === "n/a" ? "na" : "frozen";

/** Two lines at most: the block names are short phrases ("Phần nhìn ảnh"), split at the space nearest the middle. */
export function lines(text: string): string[] {
  if (text.length <= 10 || !text.includes(" ")) return [text];
  const mid = text.length / 2;
  let best = -1;
  for (let i = 0; i < text.length; i++)
    if (text[i] === " " && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  return [text.slice(0, best), text.slice(best + 1)];
}

/** "Giai đoạn 1: Phần nhìn ảnh đóng băng; Bộ nối train toàn bộ; …", then the side blocks (full names). */
export function archLabel(arch: TrainArch, tr: Translate): string {
  const stages = arch.stages?.length ? arch.stages : [arch];
  const parts = stages.map((s, i) =>
    tr("arch.aria_stage", {
      n: i + 1,
      blocks: BLOCKS.map((b) =>
        tr("arch.aria_block", {
          block: tr(`arch.block.${b}`),
          state: tr(`arch.state.${stateKey(s[b])}`),
        }),
      ).join("; "),
    }),
  );
  const extras = (arch.extras ?? []).map((e) => tr(`arch.extra.${e}`));
  return [...parts, ...(extras.length ? [`${extras.join(", ")}.`] : [])].join(" ");
}

function block(
  x: number,
  y: number,
  name: string,
  state: BlockState,
  quant: string | null,
  w = BW,
  h = BH,
): { cls: string; shapes: ArchShape[] } {
  const text = name ? lines(name) : [];
  const shapes: ArchShape[] = [{ kind: "rect", x, y, w, h, rx: 4 }];
  if (state === "frozen" || state === "lora")
    shapes.push({ kind: "icon", name: "lock", x: x + 3, y: y + 3, size: 10, cls: "arch-lock" });
  if (text.length)
    shapes.push({
      kind: "text",
      x: x + w / 2,
      y: y + h / 2 + (text.length === 1 ? 4 : -2),
      lines: text,
      dy: 12,
      anchor: "middle",
    });
  if (state === "full_low_lr")
    shapes.push({
      kind: "text",
      x: x + w - 2,
      y: y - 3,
      lines: ["LR↓"],
      dy: 0,
      anchor: "end",
      cls: "arch-mark",
    });
  if (state === "lora") {
    shapes.push({
      kind: "rect",
      x: x + w - 6,
      y: y + h / 2 - 7,
      w: 28,
      h: 14,
      rx: 7,
      cls: "arch-pill-lora",
    });
    shapes.push({
      kind: "text",
      x: x + w + 8,
      y: y + h / 2 + 3.5,
      lines: ["LoRA"],
      dy: 0,
      anchor: "middle",
      cls: "arch-pill-lora-text",
    });
  }
  if (quant) {
    shapes.push({
      kind: "rect",
      x: x + 2,
      y: y - 8,
      w: quant.length * 6 + 8,
      h: 13,
      rx: 3,
      cls: "arch-pill-quant",
    });
    shapes.push({
      kind: "text",
      x: x + 6,
      y: y + 1.5,
      lines: [quant],
      dy: 0,
      cls: "arch-pill-quant-text",
    });
  }
  return { cls: `arch-block ${blockCls(state)}`, shapes };
}

function side(
  x: number,
  y: number,
  extra: Extra,
  tr: Translate,
): { cls: string; shapes: ArchShape[] } {
  const icon =
    extra === "teacher"
      ? "brain-circuit"
      : extra === "rollout"
        ? "play"
        : extra === "reward"
          ? "list-checks"
          : null;
  const ghost = extra === "reference" || extra === "reference_shared";
  // the block says the short name; the full one ("…dùng chung model gốc…") is in the diagram's label
  const text = lines(tr(`arch.extra.${extra === "reference_shared" ? "reference" : extra}`));
  const h = SIDE_H - 6;
  const tx = x + (icon ? 18 : 6);
  const shapes: ArchShape[] = [{ kind: "rect", x, y, w: BW, h, rx: 4 }];
  if (icon) shapes.push({ kind: "icon", name: icon, x: x + 4, y: y + h / 2 - 6, size: 12 });
  shapes.push({
    kind: "text",
    x: tx,
    y: y + h / 2 + (text.length === 1 ? 3.5 : -2.5),
    lines: text,
    dy: 11,
  });
  return { cls: `arch-side${ghost ? " ghost" : ""}`, shapes };
}

/** Per stage vision → projector → language model, the quantization on the LM, side blocks in a row below. */
export function archLayout(arch: TrainArch, tr: Translate): ArchLayout {
  const stages = arch.stages?.length ? arch.stages : [arch];
  const multi = stages.length > 1;
  const extras = arch.extras ?? [];
  const x0 = multi ? LABEL_W : 0;
  const bx = (i: number) => x0 + i * (BW + GAP);
  const width = x0 + 3 * BW + 2 * GAP + 24; // room for a LoRA pill past the last block
  const sideY = PAD_TOP + stages.length * ROW_H - 10;
  const height = sideY + (extras.length ? SIDE_H : 0) + 2;
  // the side row (spec I §2, kept inside 280px): the reference under the language model; teacher, rollout and reward
  // from the left, the teacher with an arrow up into the vision block ("sinh dữ liệu")
  const left = extras.filter((e) => e !== "reference" && e !== "reference_shared");
  const sideX = (e: Extra) =>
    e === "reference" || e === "reference_shared" ? bx(2) : bx(left.indexOf(e));

  const groups: ArchLayout["groups"] = [];
  stages.forEach((s, r) => {
    const y = PAD_TOP + r * ROW_H;
    const wires: ArchShape[] = [0, 1].map((i) => ({
      kind: "wire",
      x1: bx(i) + BW,
      y1: y + BH / 2,
      x2: bx(i + 1) - 1,
      y2: y + BH / 2,
    }));
    if (multi)
      wires.push({
        kind: "text",
        x: 0,
        y: y + BH / 2 + 4,
        lines: [tr("arch.stage", { n: r + 1 })],
        dy: 0,
        cls: "arch-stage",
      });
    groups.push({ cls: "arch-row", shapes: wires });
    const quant = s.quant
      ? s.fake_quant
        ? `${tr(`arch.quant.${s.quant}`)} ${tr("arch.fake_quant")}`
        : tr(`arch.quant.${s.quant}`)
      : null;
    BLOCKS.forEach((b, i) => {
      groups.push(block(bx(i), y, tr(`arch.block.${b}`), s[b], b === "lm" ? quant : null));
    });
  });
  if (extras.includes("teacher"))
    groups.push({
      cls: "arch-row",
      shapes: [
        {
          kind: "wire",
          x1: bx(0) + BW / 2,
          y1: sideY,
          x2: bx(0) + BW / 2,
          y2: PAD_TOP + (stages.length - 1) * ROW_H + BH + 1,
        },
      ],
    });
  for (const e of extras) groups.push(side(sideX(e), sideY, e, tr));
  return { width, height, label: archLabel(arch, tr), groups };
}

/** The legend's four samples (spec I §2 "?"), each a small block without a name, and the key of its words. */
export const LEGEND_STATES: BlockState[] = ["frozen", "full", "full_low_lr", "lora"];
export function legendSample(state: BlockState): { cls: string; shapes: ArchShape[] } {
  return block(0, 0, "", state, null, 34, 20);
}
