// Print name@version + license for every installed package (from package-lock.json) so bao-security can
// fill configs/security/licenses.yaml (kind: code). Fails on licenses outside the allowlist.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const lock = JSON.parse(readFileSync(resolve(here, "../package-lock.json"), "utf8"));
const ALLOWED = new Set([
  "MIT",
  "ISC",
  "Apache-2.0",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "0BSD",
  "MIT OR Apache-2.0",
  "MPL-2.0",
  "BlueOak-1.0.0",
  "CC0-1.0",
  "MIT-0",
  "Python-2.0",
]);

const rows = [];
let bad = 0;
for (const [path, pkg] of Object.entries(lock.packages ?? {})) {
  if (!path) continue;
  const name = path.slice(path.lastIndexOf("node_modules/") + "node_modules/".length);
  const license = pkg.license ?? "UNKNOWN";
  const ok = ALLOWED.has(license);
  if (!ok) bad++;
  rows.push(`${ok ? " " : "!"} ${name}@${pkg.version}\t${license}\t${pkg.dev ? "dev" : "runtime"}`);
}
console.log(rows.sort().join("\n"));
console.log(`\n${rows.length} packages, ${bad} outside allowlist`);
process.exit(bad ? 1 : 0);
