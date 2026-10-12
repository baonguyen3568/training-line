// Convert the single i18n source (src/vlm_pipeline/i18n/{vi,en}.yaml, owner bao-design) to JSON
// bundles for the SPA. Output: src/generated/i18n.{vi,en}.json (generated, git-ignored).
// Node/error text normally comes resolved from the API; this bundle covers UI chrome and mock mode.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, "../../src/vlm_pipeline/i18n");
const outDir = resolve(here, "../src/generated");

mkdirSync(outDir, { recursive: true });
for (const locale of ["vi", "en"]) {
  const data = parse(readFileSync(resolve(srcDir, `${locale}.yaml`), "utf8"));
  writeFileSync(resolve(outDir, `i18n.${locale}.json`), JSON.stringify(data));
  console.log(`i18n: ${locale} → src/generated/i18n.${locale}.json`);
}
