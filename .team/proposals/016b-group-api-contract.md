# 016b — Group (subgraph) API contract for the UI (P016 §2, P4-4)
bao-frontend + bao-cleancode · 2026-10-14. Source of truth for the shape; chat messages crossed several times, this file wins.

## 1. Where groups come from
**Chosen (final, cleancode implements this):** `GET /api/node-types` gains `groups[]` next to `node_types[]`;
`node_types` keeps only runnable node types. A group entry has the NodeSpec field names except `group_id` (instead of
`type_id`), `title_key`/`desc_key` beside the rendered `title`/`desc`, and `body` (the definition). The frontend adapts each
entry into its spec map (type id = `group_id`).

The adapter also accepts the alternative (group entries inside `node_types[]` with a `group` field), so a later change of
mind on the server needs no frontend change.

## 2. Fields of a group entry
| Field | Value |
|---|---|
| `group_id` | `group.<slug>` |
| `version`, `title_key`, `desc_key`, `title`, `desc` | as nodes; title/desc rendered from `groups.<slug>.title/desc` for `?locale=` |
| `category` | the lane of the inner node producing the FIRST exposed output (LoRA group → train); no output → first inner node. Never "group". |
| `icon` | omitted; the UI always draws Lucide `layers` |
| `inputs` | `{name: {type, optional, many}}`; every target of one exposed input has the SAME type (checked when the group loads); `optional` only when every target is optional |
| `outputs` | `{name: {type, optional: false, many: false}}` |
| `params_schema` | `{type: "object", additionalProperties: false, properties: {name: <inner param's property, x-ui kept; default = the group's preset if any>}}` |
| `cacheable` / `runner` | display only (a group never runs itself): cacheable = all inner cacheable; runner = "container" if any inner container step |
| `permissions` | union of the inner steps (same shape as nodes) |
| `origin` | `{kind: "first_party", trust_min}`: lowest trust of the inner steps; a group never raises trust |
| `available`, `unavailable_reason` | false + the first blocked inner step's reason (with `inner_type`) when any inner step has no worker ; it reflects worker support only, NOT param/model refusals of check_support (e.g. GRPO single-GPU): those come from validate, so `available: true` does not promise a clean run |
| `policy_block` | the first blocked inner step's, else null |
| `body` | `{nodes, edges, inputs, outputs, params}`: exactly the GroupDefinition body, for the read-only inside view (first-party groups are always read-only) |

## 3. Validate and run
- An issue on an inner step: `node_id` = the OUTERMOST group node, `inner_path` = ids per level; other fields
  (port/param) are the inner step's. The group's own problem is CFG-012 on the group node.
- Run events use the flat id `<group>__<inner>` (recursive); the part before the FIRST `__` is the outermost group.
  User ids never contain `__` (CFG-007).

## 4. User groups (`user.*`) inside a workflow file
The frontend derives the same entry from the file's GroupDefinition with the rules of §2 (`read_only` is true in 1.0 too:
design 818e7c3). Both sides test against golden fixtures `schemas/group_examples/*.json` (definition → derived entry,
incl. nesting and a type mismatch) exported by cleancode; server validation still expands and reports real wiring
errors if the UI derived wrongly.
