// The frontend derives user.* groups itself (016b §4); it must agree with the server on cleancode's golden fixtures
// (schemas/group_examples, bundled by `npm run schemas`): same ports, params (presets as defaults) and lane.
import { describe, expect, it } from "vitest";
import type { GroupDefinition, NodeSpec, PortSpec } from "../src/api/types";
import golden from "../src/generated/group-examples.json";
import nodeTypes from "../src/generated/node-types.json";
import { deriveGroupSpec } from "../src/lib/groups";

interface Example {
  group: GroupDefinition;
  uses: GroupDefinition[];
  expected: Record<string, unknown> & { error?: string };
}
const examples = golden as unknown as Record<string, Example>;
const base = new Map<string, NodeSpec>(
  (nodeTypes as unknown as { node_types: NodeSpec[] }).node_types.map((s) => [s.type_id, s]),
);
const ports = (p: Record<string, PortSpec>) =>
  Object.fromEntries(
    Object.entries(p).map(([k, v]) => [
      k,
      { type: v.type, optional: !!v.optional, many: !!v.many },
    ]),
  );

describe("group derivation matches the server (golden fixtures)", () => {
  for (const [name, ex] of Object.entries(examples)) {
    it(name, () => {
      const specs = new Map(base);
      for (const used of ex.uses) {
        const s = deriveGroupSpec(used, specs);
        if (s) specs.set(used.id, s);
      }
      const spec = deriveGroupSpec(ex.group, specs);
      if (ex.expected.error) {
        expect(spec).toBeNull(); // the server refuses it too (message in the fixture)
        return;
      }
      if (!spec) throw new Error(`could not derive ${ex.group.id}`);
      const want = ex.expected as unknown as NodeSpec & { group_id: string };
      expect(spec.type_id).toBe(want.group_id);
      expect(spec.category).toBe(want.category);
      expect(ports(spec.inputs)).toEqual(ports(want.inputs));
      expect(ports(spec.outputs)).toEqual(ports(want.outputs));
      expect(spec.params_schema.properties).toEqual(want.params_schema.properties);
      if ("runner" in want) expect(spec.runner).toBe(want.runner);
      if ("cacheable" in want) expect(spec.cacheable).toBe(want.cacheable);
    });
  }
});
