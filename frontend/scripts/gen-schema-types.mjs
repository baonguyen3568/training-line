// Generate TypeScript types from the backend's JSON Schemas (schemas/*.schema.json, source of truth;
// P004 §8.2). Dependency-free: supports the subset pydantic emits for our models (type, enum, anyOf,
// $ref → $defs, arrays, objects with required/additionalProperties). Output is committed;
// `npm run check` regenerates it and fails on drift (git diff --exit-code).
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const schemaDir = resolve(here, "../../schemas");
const outFile = resolve(here, "../src/generated/schemas.ts");

const prim = {
  string: "string",
  number: "number",
  integer: "number",
  boolean: "boolean",
  null: "null",
};
const ident = (s) => s.replace(/[^A-Za-z0-9_]/g, "_");
const key = (k) => (/^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k));

function ts(s, indent = "") {
  if (s === true || s === undefined || (typeof s === "object" && Object.keys(s).length === 0))
    return "unknown";
  if (s.$ref) return ident(s.$ref.split("/").pop());
  if (s.enum) return s.enum.map((v) => JSON.stringify(v)).join(" | ");
  if (s.const !== undefined) return JSON.stringify(s.const);
  if (s.anyOf || s.oneOf)
    return [...new Set((s.anyOf ?? s.oneOf).map((x) => ts(x, indent)))].join(" | ");
  if (Array.isArray(s.type)) return s.type.map((t) => ts({ ...s, type: t }, indent)).join(" | ");
  if (s.type === "array") {
    const item = ts(s.items, indent);
    return /[|&]/.test(item) ? `(${item})[]` : `${item}[]`;
  }
  if (s.type === "object" || s.properties) return objectType(s, indent);
  if (s.type in prim) return prim[s.type];
  throw new Error(`unsupported schema: ${JSON.stringify(s).slice(0, 120)}`);
}

function objectType(s, indent) {
  const inner = `${indent}  `;
  const req = new Set(s.required ?? []);
  const lines = Object.entries(s.properties ?? {}).map(([k, v]) => {
    const doc = v.description ? `${inner}/** ${v.description.replace(/\*\//g, "*\\/")} */\n` : "";
    return `${doc}${inner}${key(k)}${req.has(k) ? "" : "?"}: ${ts(v, inner)};`;
  });
  const ap = s.additionalProperties;
  if (ap && ap !== false)
    lines.push(
      `${inner}[extra: string]: ${ap === true ? "unknown" : `${ts(ap, inner)} | undefined`};`,
    );
  return lines.length ? `{\n${lines.join("\n")}\n${indent}}` : "Record<string, never>";
}

const files = readdirSync(schemaDir)
  .filter((f) => f.endsWith(".schema.json"))
  .sort();
const out = [
  "// GENERATED — do not edit. Source: schemas/*.schema.json (backend). Regenerate: node scripts/gen-schema-types.mjs",
  "/* biome-ignore-all lint: generated file */",
  "",
];
const emitted = new Set();
for (const f of files) {
  const schema = JSON.parse(readFileSync(resolve(schemaDir, f), "utf8"));
  for (const [name, def] of Object.entries(schema.$defs ?? {})) {
    if (emitted.has(name)) continue;
    emitted.add(name);
    out.push(`export type ${ident(name)} = ${ts(def)};`, "");
  }
  const name = ident(schema.title ?? basename(f, ".schema.json"));
  if (emitted.has(name)) continue;
  emitted.add(name);
  if (schema.description) out.push(`/** ${schema.description.split("\n")[0]} */`);
  out.push(`export type ${name} = ${ts(schema)};`, "");
}
writeFileSync(outFile, out.join("\n"));

// Registry snapshot (REGISTRY.to_api(), exported by the backend) → mock API node types, so the mock never
// drifts from the real nodes (design review M1 r3). Byte-identical copy; committed and drift-checked.
const snapshot = readFileSync(resolve(schemaDir, "node_types.snapshot.json"), "utf8");
writeFileSync(resolve(here, "../src/generated/node-types.json"), snapshot);

// Templates: single source configs/templates/<id>.json (backend serves them at /api/templates). Bundled
// for the mock API in id order; estimates are computed by the server from the catalog, not stored here.
const tplDir = resolve(here, "../../configs/templates");
const templates = readdirSync(tplDir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => JSON.parse(readFileSync(resolve(tplDir, f), "utf8")));
writeFileSync(
  resolve(here, "../src/generated/templates.json"),
  `${JSON.stringify(templates, null, 2)}\n`,
);
console.log(`schemas: ${emitted.size} types → src/generated/schemas.ts`);
