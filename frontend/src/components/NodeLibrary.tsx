import { type CSSProperties, useMemo, useRef, useState } from "react";
import type { NodeSpec } from "../api/types";
import { t, tp } from "../i18n";
import { LANES, type Lane, laneOf } from "../lib/lanes";
import { nodeDesc, nodeTitle } from "../lib/nodeText";
import { portShape, tokenVar } from "../lib/ports";
import { expandQuery, matches } from "../lib/search";
import { useWorkflow } from "../store/workflow";
import { Icon } from "./Icon";

export const DRAG_MIME = "application/vlm-node-type";

/** Type being dragged from the library: dataTransfer data is unreadable during dragover, the canvas needs it to
 * open the right (compact) lane under the cursor (design M2-6 r1 B3). */
export const libraryDrag: { typeId: string | null } = { typeId: null };

/** Left sidebar (M1-4, design spec M1-5): grouped by lane, accent-insensitive search, drag or double-click/Enter. */
export function NodeLibrary({
  onAdd,
  disabled,
}: {
  onAdd: (typeId: string) => void;
  disabled: boolean;
}) {
  const specs = useWorkflow((s) => s.specs);
  const ports = useWorkflow((s) => s.ports);
  const portColor = (type: string) => {
    const pt = ports.find((p) => p.id === type);
    return pt ? tokenVar(pt.color_token) : "var(--color-port-value)";
  };
  const portName = (type: string) => {
    const pt = ports.find((p) => p.id === type);
    return pt ? tp(pt.label_key.split(".")) : type;
  };
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const groups = useMemo(() => {
    const terms = expandQuery(query);
    const byLane = new Map<Lane, NodeSpec[]>();
    for (const spec of specs.values()) {
      const text = `${nodeTitle(spec, spec.type_id)} ${nodeDesc(spec, spec.type_id)} ${spec.type_id}`;
      if (!matches(text, terms)) continue;
      const lane = laneOf(spec.category);
      byLane.set(lane, [...(byLane.get(lane) ?? []), spec]);
    }
    return LANES.filter((l) => byLane.has(l)).map((l) => [l, byLane.get(l) ?? []] as const);
  }, [specs, query]);

  if (collapsed) {
    return (
      <aside className="library collapsed">
        <button
          type="button"
          className="link"
          onClick={() => setCollapsed(false)}
          aria-label={t("library.expand")}
        >
          ☰
        </button>
      </aside>
    );
  }

  return (
    <aside
      className="library"
      aria-label={t("library.title")}
      onKeyDown={(e) => {
        if (e.key === "/" && document.activeElement !== input.current) {
          e.preventDefault();
          input.current?.focus();
        }
      }}
    >
      <div className="library-head">
        <strong>{t("library.title")}</strong>
        <button
          type="button"
          className="link"
          onClick={() => setCollapsed(true)}
          aria-label={t("library.collapse")}
        >
          ⟨
        </button>
      </div>
      <input
        ref={input}
        type="search"
        placeholder={t("library.search")}
        aria-label={t("library.search")}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {groups.length === 0 && <p className="muted small">{t("library.no_match")}</p>}
      {groups.map(([lane, items]) => (
        <section key={lane} className="library-group">
          <h3>{t(`canvas.lane.${lane}`)}</h3>
          <ul>
            {items.map((spec) => {
              const blocked = !!spec.policy_block || disabled;
              const out = Object.values(spec.outputs)[0];
              return (
                <li key={spec.type_id}>
                  <button
                    type="button"
                    className={`library-item${spec.origin.kind === "third_party" ? " third-party" : ""}`}
                    data-testid={`library-item-${spec.type_id}`}
                    draggable={!blocked}
                    disabled={blocked}
                    title={nodeDesc(spec, spec.type_id)}
                    onDragStart={(e) => {
                      e.dataTransfer.setData(DRAG_MIME, spec.type_id);
                      e.dataTransfer.effectAllowed = "copy";
                      libraryDrag.typeId = spec.type_id;
                      window.dispatchEvent(
                        new CustomEvent("vlm:library-drag", { detail: spec.type_id }),
                      );
                    }}
                    onDragEnd={() => {
                      libraryDrag.typeId = null;
                      window.dispatchEvent(new CustomEvent("vlm:library-drag", { detail: null }));
                    }}
                    onDoubleClick={() => onAdd(spec.type_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") onAdd(spec.type_id);
                    }}
                  >
                    <span className="library-title">
                      {out && (
                        // what this step produces: port shape + colour (+ the name in the tooltip/label)
                        <span
                          className={`out-port pg pg-${portShape(out.type)}`}
                          style={{ "--pc": portColor(out.type) } as CSSProperties}
                          title={portName(out.type)}
                          aria-hidden="true"
                        />
                      )}
                      {spec.origin.kind === "third_party" && "🧩 "}
                      {nodeTitle(spec, spec.type_id)}
                    </span>
                    <span className="library-desc">{nodeDesc(spec, spec.type_id)}</span>
                    {spec.policy_block && (
                      <span className="badge blocked">
                        <Icon name="ban" size={12} /> {t("library.blocked")}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </aside>
  );
}
