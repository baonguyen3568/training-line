import type { PortType } from "../api/types";

/** An output of type `out` may feed an input of type `input` if `input` is `out` or one of its ancestors. */
export function isCompatible(out: string, input: string, ports: readonly PortType[]): boolean {
  const parent = new Map(ports.map((p) => [p.id, p.parent]));
  let cur: string | null | undefined = out;
  for (let guard = 0; cur && guard < 16; guard++) {
    if (cur === input) return true;
    cur = parent.get(cur);
  }
  return false;
}

/** "color.port.model" → "var(--color-port-model)" (matches scripts/build-tokens.mjs naming). */
export function tokenVar(token: string): string {
  return `var(--${token.replaceAll(".", "-").replaceAll("_", "-")})`;
}
