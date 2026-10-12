import type { NodeSpec, ParamSchema } from "../api/types";

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Advanced params whose value differs from the default — surfaced in Simple as "⚙ n tuỳ chỉnh" (P010 §10). */
export function customizedAdvanced(
  spec: NodeSpec | undefined,
  params: Record<string, unknown>,
): string[] {
  if (!spec) return [];
  return Object.entries(spec.params_schema.properties)
    .filter(
      ([name, s]: [string, ParamSchema]) =>
        s["x-ui"]?.advanced && name in params && !same(params[name], s.default),
    )
    .map(([name]) => name);
}
