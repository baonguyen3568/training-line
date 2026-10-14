import { useId } from "react";
import { t, tp } from "../i18n";
import {
  type ArchShape,
  archLayout,
  LEGEND_STATES,
  legendSample,
  type TrainArch,
  type Translate,
} from "../lib/archLayout";
import { Icon } from "./Icon";

export type { TrainArch } from "../lib/archLayout";

const tr: Translate = (key, params) => (params ? tp(key.split("."), params) : t(key));

function Shape({ s, marker }: { s: ArchShape; marker: string }) {
  switch (s.kind) {
    case "rect":
      return <rect className={s.cls} x={s.x} y={s.y} width={s.w} height={s.h} rx={s.rx} />;
    case "icon":
      return (
        <g className={s.cls} transform={`translate(${s.x} ${s.y})`}>
          <Icon name={s.name} size={s.size} />
        </g>
      );
    case "wire":
      return (
        <line
          className="arch-wire"
          x1={s.x1}
          y1={s.y1}
          x2={s.x2}
          y2={s.y2}
          markerEnd={marker ? `url(#${marker})` : undefined}
        />
      );
    case "text":
      return (
        <text className={s.cls} x={s.x} y={s.y} textAnchor={s.anchor}>
          {s.lines.map((l, i) => (
            <tspan key={l} x={s.x} dy={i === 0 ? 0 : s.dy}>
              {l}
            </tspan>
          ))}
        </text>
      );
  }
}

/**
 * The training architecture of a method (design spec I §2), drawn from lib/archLayout (shared with the site): per
 * stage vision → projector → language model, each block's state never colour-only (dashed + lock = frozen, filled =
 * trained, "LR↓", a "LoRA" pill), the quantization as a pill, side blocks below. One `role="img"` named from the data.
 */
export function ArchDiagram({ arch }: { arch: TrainArch }) {
  const marker = useId().replace(/:/g, "");
  const layout = archLayout(arch, tr);
  return (
    <svg
      className="arch-diagram"
      role="img"
      aria-label={layout.label}
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      width="100%"
      style={{ maxWidth: layout.width }}
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
      {layout.groups.map((g, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: the layout is a fixed ordered list
        <g key={i} className={g.cls}>
          {g.shapes.map((s, j) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the layout is a fixed ordered list
            <Shape key={j} s={s} marker={marker} />
          ))}
        </g>
      ))}
    </svg>
  );
}

/** The "?" legend of the diagram (spec I §2): the four block states, drawn small, each with its words. */
export function ArchLegend() {
  return (
    <ul className="arch-legend">
      {LEGEND_STATES.map((state) => {
        const g = legendSample(state);
        return (
          <li key={state}>
            <svg
              className="arch-diagram"
              width={58}
              height={22}
              viewBox="-1 -9 58 30"
              aria-hidden="true"
            >
              <g className={g.cls}>
                {g.shapes.map((s, j) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: the layout is a fixed ordered list
                  <Shape key={j} s={s} marker="" />
                ))}
              </g>
            </svg>
            <span>{t(`arch.state.${state}`)}</span>
          </li>
        );
      })}
    </ul>
  );
}
