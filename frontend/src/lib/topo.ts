import type { Workflow } from "../api/types";

/** Step numbers ①②③ by topological order (P010 §2); nodes in a cycle get no number (0). */
export function stepNumbers(wf: Workflow): Map<string, number> {
  const indeg = new Map(wf.nodes.map((n) => [n.id, 0]));
  const next = new Map<string, string[]>(wf.nodes.map((n) => [n.id, []]));
  for (const e of wf.edges) {
    if (!indeg.has(e.from[0]) || !indeg.has(e.to[0])) continue;
    indeg.set(e.to[0], (indeg.get(e.to[0]) ?? 0) + 1);
    next.get(e.from[0])?.push(e.to[0]);
  }
  const order = new Map<string, number>();
  let frontier = wf.nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  let step = 1;
  while (frontier.length) {
    const nextFrontier: string[] = [];
    for (const id of frontier) {
      order.set(id, step++);
      for (const m of next.get(id) ?? []) {
        const d = (indeg.get(m) ?? 0) - 1;
        indeg.set(m, d);
        if (d === 0) nextFrontier.push(m);
      }
    }
    frontier = nextFrontier;
  }
  for (const n of wf.nodes) if (!order.has(n.id)) order.set(n.id, 0);
  return order;
}
