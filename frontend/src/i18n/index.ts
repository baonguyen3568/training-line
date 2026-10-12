// UI copy lookup over the bundles generated from src/vlm_pipeline/i18n/{vi,en}.yaml (owner: bao-design).
// Rules (design/copy/README.md): missing key → show the key; node type ids contain dots and are ONE key
// under `node`, so lookups take an explicit path array instead of splitting on dots.
import en from "../generated/i18n.en.json";
import vi from "../generated/i18n.vi.json";

export type Locale = "vi" | "en";
type Tree = { [key: string]: Tree | string };

const bundles: Record<Locale, Tree> = { vi: vi as Tree, en: en as Tree };
let current: Locale = "vi";

export function setLocale(locale: Locale): void {
  current = locale;
}
export function getLocale(): Locale {
  return current;
}

export type Params = Record<string, string | number>;

function lookup(path: readonly string[], locale: Locale): string | undefined {
  let node: Tree | string | undefined = bundles[locale];
  for (const part of path) {
    if (node === undefined || typeof node === "string") return undefined;
    node = node[part];
  }
  return typeof node === "string" ? node : undefined;
}

function interpolate(template: string, params?: Params): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in params ? String(params[name]) : match,
  );
}

/** Translate a dotted key whose segments contain no dots, e.g. "gallery.title". */
export function t(key: string, params?: Params): string {
  return tp(key.split("."), params);
}

/** Translate an explicit path, e.g. ["node", "train.lora_vlm", "title"]. Falls back to the joined key. */
export function tp(path: readonly string[], params?: Params): string {
  const found = lookup(path, current);
  return found === undefined ? path.join(".") : interpolate(found, params);
}

/** Like tp but returns undefined when missing, for optional copy such as variants. */
export function tpOptional(path: readonly string[], params?: Params): string | undefined {
  const found = lookup(path, current);
  return found === undefined ? undefined : interpolate(found, params);
}
