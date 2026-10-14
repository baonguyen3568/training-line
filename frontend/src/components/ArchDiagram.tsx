import { useId } from "react";
import { t, tp } from "../i18n";
import { Icon } from "./Icon";

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

const BLOCKS = ["vision", "projector", "lm"] as const;
type Block = (typeof BLOCKS)[number];

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

/** Two lines at most: the block names are short phrases ("Phần nhìn ảnh"), split at the space nearest the middle. */
function lines(text: string): string[] {
  if (text.length <= 10 || !text.includes(" ")) return [text];
  const mid = text.length / 2;
  let best = -1;
  for (let i = 0; i < text.length; i++)
    if (text[i] === " " && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  return [text.slice(0, best), text.slice(best + 1)];
}

function quantLabel(stage: ArchStage): string | null {
  if (!stage.quant) return null;
  const q = t(`arch.quant.${stage.quant}`);
  return stage.fake_quant ? `${q} ${t("arch.fake_quant")}` : q;
}

/** "Giai đoạn 1: Phần nhìn ảnh đóng băng; Bộ nối train toàn bộ; …" (spec I §2 a11y), generated from the data. */
export function archAriaLabel(arch: TrainArch): string {
  const stages = arch.stages?.length ? arch.stages : [arch];
  const parts = stages.map((s, i) =>
    tp(["arch", "aria_stage"], {
      n: i + 1,
      blocks: BLOCKS.map((b) =>
        tp(["arch", "aria_block"], {
          block: t(`arch.block.${b}`),
          state: t(`arch.state.${stateKey(s[b])}`),
        }),
      ).join("; "),
    }),
  );
  const extras = (arch.extras ?? []).map((e) => t(`arch.extra.${e}`));
  return [...parts, ...(extras.length ? [`${extras.join(", ")}.`] : [])].join(" ");
}

function BlockShape({
  x,
  y,
  block,
  state,
  quant,
}: {
  x: number;
  y: number;
  block: Block;
  state: BlockState;
  quant: string | null;
}) {
  const text = lines(t(`arch.block.${block}`));
  const cls =
    state === "full"
      ? "train"
      : state === "full_low_lr"
        ? "train-low"
        : state === "n/a"
          ? "na"
          : "frozen";
  return (
    <g className={`arch-block ${cls}`}>
      <rect x={x} y={y} width={BW} height={BH} rx={4} />
      {(state === "frozen" || state === "lora") && (
        <g transform={`translate(${x + 4} ${y + 4})`} className="arch-lock">
          <Icon name="lock" size={10} />
        </g>
      )}
      <text x={x + BW / 2} y={y + BH / 2 + (text.length === 1 ? 4 : -2)} textAnchor="middle">
        {text.map((l, i) => (
          <tspan key={l} x={x + BW / 2} dy={i === 0 ? 0 : 12}>
            {l}
          </tspan>
        ))}
      </text>
      {state === "full_low_lr" && (
        <text className="arch-mark" x={x + BW - 2} y={y - 3} textAnchor="end">
          LR↓
        </text>
      )}
      {state === "lora" && (
        <g className="arch-pill lora">
          <rect x={x + BW - 6} y={y + BH / 2 - 7} width={28} height={14} rx={7} />
          <text x={x + BW + 8} y={y + BH / 2 + 3.5} textAnchor="middle">
            LoRA
          </text>
        </g>
      )}
      {quant && (
        <g className="arch-pill quant">
          <rect x={x + 2} y={y - 8} width={quant.length * 6 + 8} height={13} rx={3} />
          <text x={x + 6} y={y + 1.5}>
            {quant}
          </text>
        </g>
      )}
    </g>
  );
}

function SideBlock({ x, y, extra }: { x: number; y: number; extra: Extra }) {
  const icon =
    extra === "teacher"
      ? "brain-circuit"
      : extra === "rollout"
        ? "play"
        : extra === "reward"
          ? "list-checks"
          : null;
  const ghost = extra === "reference" || extra === "reference_shared";
  const label = extra === "reference_shared" ? t("arch.extra.reference") : t(`arch.extra.${extra}`);
  const text = lines(label);
  const w = BW;
  const h = SIDE_H - 6;
  const tx = x + (icon ? 18 : 6);
  return (
    <g className={`arch-side${ghost ? " ghost" : ""}`}>
      <rect x={x} y={y} width={w} height={h} rx={4} />
      {icon && (
        <g transform={`translate(${x + 4} ${y + h / 2 - 6})`}>
          <Icon name={icon} size={12} />
        </g>
      )}
      <text x={tx} y={y + h / 2 + (text.length === 1 ? 3.5 : -2.5)}>
        {text.map((l, i) => (
          <tspan key={l} x={tx} dy={i === 0 ? 0 : 11}>
            {l}
          </tspan>
        ))}
      </text>
    </g>
  );
}

/**
 * The training architecture of a method (design spec I §2): per stage, vision → projector → language model, each
 * block's state drawn without relying on colour (dashed + lock = frozen, filled = trained, "LR↓", a "LoRA" pill), the
 * quantization as a pill, side blocks for teacher / reference / rollout / reward. One `role="img"` with a label built
 * from the data; the drawing itself is hidden from assistive tech.
 */
export function ArchDiagram({ arch }: { arch: TrainArch }) {
  const marker = useId().replace(/:/g, "");
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

  return (
    <svg
      className="arch-diagram"
      role="img"
      aria-label={archAriaLabel(arch)}
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ maxWidth: width }}
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 6 6"
          refX="5"
          refY="3"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0 0 6 3 0 6z" className="arch-arrowhead" />
        </marker>
      </defs>
      {/* role="img": the children are presentational, the label above says it all */}
      <g>
        {stages.map((s, r) => {
          const y = PAD_TOP + r * ROW_H;
          const q = quantLabel(s);
          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: stages are a fixed ordered list
            <g key={r}>
              {multi && (
                <text className="arch-stage" x={0} y={y + BH / 2 + 4}>
                  {t("arch.stage", { n: r + 1 })}
                </text>
              )}
              {[0, 1].map((i) => (
                <line
                  key={i}
                  className="arch-wire"
                  x1={bx(i) + BW}
                  y1={y + BH / 2}
                  x2={bx(i + 1) - 1}
                  y2={y + BH / 2}
                  markerEnd={`url(#${marker})`}
                />
              ))}
              {BLOCKS.map((b, i) => (
                <BlockShape
                  key={b}
                  x={bx(i)}
                  y={y}
                  block={b}
                  state={s[b]}
                  quant={b === "lm" ? q : null}
                />
              ))}
            </g>
          );
        })}
        {extras.includes("teacher") && (
          <line
            className="arch-wire"
            x1={bx(0) + BW / 2}
            y1={sideY}
            x2={bx(0) + BW / 2}
            y2={PAD_TOP + (stages.length - 1) * ROW_H + BH + 1}
            markerEnd={`url(#${marker})`}
          />
        )}
        {extras.map((e) => (
          <SideBlock key={e} x={sideX(e)} y={sideY} extra={e} />
        ))}
      </g>
    </svg>
  );
}

const LEGEND: { state: BlockState; key: string }[] = [
  { state: "frozen", key: "frozen" },
  { state: "full", key: "full" },
  { state: "full_low_lr", key: "full_low_lr" },
  { state: "lora", key: "lora" },
];

/** The "?" legend of the diagram (spec I §2): the four block states, drawn small, each with its words. */
export function ArchLegend() {
  return (
    <ul className="arch-legend">
      {LEGEND.map(({ state, key }) => {
        const cls = state === "full" ? "train" : state === "full_low_lr" ? "train-low" : "frozen";
        return (
          <li key={key}>
            <svg
              className="arch-diagram"
              width={58}
              height={22}
              viewBox="-1 -9 58 30"
              aria-hidden="true"
            >
              <g className={`arch-block ${cls}`}>
                <rect x={0} y={0} width={34} height={20} rx={3} />
                {(state === "frozen" || state === "lora") && (
                  <g transform="translate(3 3)" className="arch-lock">
                    <Icon name="lock" size={10} />
                  </g>
                )}
                {state === "full_low_lr" && (
                  <text className="arch-mark" x={34} y={-2} textAnchor="end">
                    LR↓
                  </text>
                )}
                {state === "lora" && (
                  <g className="arch-pill lora">
                    <rect x={28} y={3} width={28} height={14} rx={7} />
                    <text x={42} y={13.5} textAnchor="middle">
                      LoRA
                    </text>
                  </g>
                )}
              </g>
            </svg>
            <span>{t(`arch.state.${key}`)}</span>
          </li>
        );
      })}
    </ul>
  );
}
