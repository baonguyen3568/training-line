import type { ReactNode } from "react";

/** Params whose values are always machine strings (file names, paths): e.g. EXP-004 {files} "config.json, …". */
const MACHINE_PARAMS = new Set(["files", "file", "path", "run_dir", "missing_files"]);
const looksMachine = (v: string) => /[/_.-]\S*\d|\//.test(v);

/**
 * A sentence with its machine strings (paths, file names, run dirs) set as <code>, in the order they appear
 * (Mono-Means-Machine): values of MACHINE_PARAMS always, other string params when they look like paths.
 */
export function withCode(text: string, params: Record<string, unknown>): ReactNode {
  const tokens = Object.entries(params)
    .filter(
      (e): e is [string, string] =>
        typeof e[1] === "string" &&
        e[1].length > 0 &&
        (MACHINE_PARAMS.has(e[0]) || looksMachine(e[1])),
    )
    .map(([, v]) => v)
    .filter((v) => text.includes(v))
    .sort((a, b) => text.indexOf(a) - text.indexOf(b));
  if (tokens.length === 0) return text;
  const parts: ReactNode[] = [];
  let rest = text;
  for (const tok of tokens) {
    const i = rest.indexOf(tok);
    if (i < 0) continue;
    parts.push(rest.slice(0, i), <code key={`${tok}-${parts.length}`}>{tok}</code>);
    rest = rest.slice(i + tok.length);
  }
  parts.push(rest);
  return parts;
}
