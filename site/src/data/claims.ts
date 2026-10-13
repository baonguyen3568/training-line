// Every number shown on marketing pages comes from claims.json, never typed into a page.
// Shape: {"<id>": {"value": "71,65", "unit": "%", "source": "benchmarks/eval/....json"}}.
// scripts/qa/check_site_numbers.py (bao-tester) checks each value appears in its source file.
import claims from "./claims.json";

export interface Claim {
  value: string;
  unit?: string;
  source: string;
}

const all: Record<string, Claim> = claims;

/** Look up a claim; an unknown id fails the build instead of rendering an empty number. */
export function claim(id: string): Claim {
  const c = all[id];
  if (!c) throw new Error(`claims.json: unknown claim "${id}"`);
  return c;
}
