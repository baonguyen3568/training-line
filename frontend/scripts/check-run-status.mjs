// Drift check: RUN_STATUSES in src/api/types.ts must equal backend `common/runinfo.RunStatus` (source of truth).
// Plain Node (no TS types needed); runs in `npm run check`. Replace with schemas/ JSON Schema once exported.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const py = readFileSync(resolve(here, "../../src/vlm_pipeline/common/runinfo.py"), "utf8");
const ts = readFileSync(resolve(here, "../src/api/types.ts"), "utf8");

const pyBlock = py.slice(py.indexOf("class RunStatus"));
const pyEnd = pyBlock.search(/\n\s*@property|\nclass |\ndef /);
const pyValues = [...pyBlock.slice(0, pyEnd).matchAll(/^\s+[A-Z_]+\s*=\s*"([a-z_]+)"/gm)].map(
  (m) => m[1],
);
const tsBlock = /RUN_STATUSES = \[([^\]]*)\]/.exec(ts)?.[1] ?? "";
const tsValues = [...tsBlock.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]);

if (pyValues.length === 0 || JSON.stringify(pyValues) !== JSON.stringify(tsValues)) {
  console.error(
    `RunStatus drift:\n  backend:  ${pyValues.join(",")}\n  frontend: ${tsValues.join(",")}`,
  );
  process.exit(1);
}
console.log(`run-status: ${tsValues.length} values in sync with common/runinfo.RunStatus`);
