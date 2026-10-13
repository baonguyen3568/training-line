import type { Workflow } from "../api/types";

/** Nearest node above `from` (walking input wires back, breadth-first) whose type matches; `from` itself first. */
export function nearestUpstream(
  wf: Workflow | null | undefined,
  from: string,
  match: (type: string) => boolean,
): string | undefined {
  if (!wf) return undefined;
  const typeOf = new Map(wf.nodes.map((n) => [n.id, n.type]));
  const seen = new Set([from]);
  let level = [from];
  while (level.length) {
    const hit = level.find((id) => match(typeOf.get(id) ?? ""));
    if (hit) return hit;
    const next: string[] = [];
    for (const id of level)
      for (const e of wf.edges)
        if (e.to[0] === id && !seen.has(e.from[0])) {
          seen.add(e.from[0]);
          next.push(e.from[0]);
        }
    level = next;
  }
  return undefined;
}

export const isModelType = (type: string) => type.startsWith("model.");
