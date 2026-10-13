import type { Connection } from "@xyflow/react";

export interface HandleRef {
  nodeId: string;
  id?: string | null;
  type: string; // "source" | "target"
}

/**
 * The React Flow handle under a drop point, read from the DOM. React Flow finds the target from its stored handle
 * bounds; right after a node mounts those can be stale and a drop exactly on a port finds nothing (tester trace
 * 2026-10-13, ~1/100), so the wire silently "does not stick". The DOM is the ground truth for what is under the
 * pointer.
 */
export function handleUnder(el: Element | null): HandleRef | null {
  const h = el?.closest(".react-flow__handle");
  const nodeId = h?.getAttribute("data-nodeid");
  if (!h || !nodeId) return null;
  const type = h.classList.contains("source")
    ? "source"
    : h.classList.contains("target")
      ? "target"
      : "";
  if (!type) return null;
  return { nodeId, id: h.getAttribute("data-handleid"), type };
}

/** source→target connection between two handles of opposite kinds, else null. */
export function connectionBetween(from: HandleRef, to: HandleRef): Connection | null {
  if (from.type === to.type || from.nodeId === to.nodeId) return null;
  const [src, dst] = from.type === "source" ? [from, to] : [to, from];
  return {
    source: src.nodeId,
    sourceHandle: src.id ?? null,
    target: dst.nodeId,
    targetHandle: dst.id ?? null,
  };
}
