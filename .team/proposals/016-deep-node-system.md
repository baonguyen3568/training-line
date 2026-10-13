# Proposal 016: Deep node system (many fine-grained nodes, subgraphs, Advanced params from the engine)

- **Author:** bao-cleancode (co-owner bao-backend). **Task:** M4 P4-3. **Status:** DRAFT, for review by backend, frontend, design and security.
- **Scope:** `nodes/`, `graph/`, the workflow file. UX (expand/collapse, library search) is P4-4 and builds on this.

## 1. Fine-grained nodes: same NodeSpec, no new framework
- One module per category (`nodes/data.py`, `train.py`, `quantize.py`, `export.py`…), explicit `REGISTRY` list as today
  (ADR-004 A3: no discovery). New nodes: data filter/dedupe/resize/convert/caption/augment/mix; train SFT/LoRA/QLoRA/DPO/GRPO;
  quantize FP8/NVFP4/AWQ/QAT; export HF/TRT-LLM.
- **A node exists in the registry only with a runner, or it is marked `available=False` with a reason code** (B-0). The
  validator reports it before the run ("this step does not run yet"), so the library can show the whole roadmap honestly.
- **Rule of three:** shared param blocks (batch, optimizer, LoRA dims…) are pydantic mixins in `<stage>/config.py`, not a
  base-node class hierarchy. Cross-node checks (model ↔ node, data format ↔ model) are `GraphRule`s (ce7f348), one per concern.

## 2. Subgraph = a group node that **expands before validation**
- A group is DATA: `{id, version, title_key, nodes[], edges[], inputs{name: [inner_node, port]}, outputs{…},
  params{name: [inner_node, param]}}` (exposed ports and params only). First-party groups live in `configs/groups/*.json`.
  A workflow that uses a user-made group embeds its definition under `groups{}` in the file (shareable, no install step).
- `graph.expand(workflow, groups) -> workflow` inlines every group instance (inner ids `<group_node>/<inner_id>`) and maps
  exposed params and wires. **Validator, cache keys and executor only see the flat graph**: no second execution model,
  and an expanded inner node caches exactly like a hand-placed one. Errors on inner nodes are re-anchored to the group node
  (`node_id`) + `inner_path` so the UI can point inside.
- Limits: depth ≤ 3, no recursion (cycle check on group ids), group versions pinned like node versions (CFG-009).
- The "big" beginner nodes (`train.lora_vlm`…) can become first-party groups later **without changing saved workflows**:
  the type id stays and resolves to a group.

## 3. Advanced params generated from the engine, through an allowlist
- A script run inside the pinned engine image (`scripts/probe/engine_schema.py`) dumps the recipe options we support
  (dotted key, type, default, bounds, doc) to `configs/engine/<engine>-<image tag>.json`, committed and reviewed.
- Each node's `params_model` gets an `advanced: AdvancedOverrides` field built from that file: **only listed keys**, typed,
  `x-ui.advanced=true`. Never a free dict: `_target_`, paths, `trust_remote_code`, and anything callable stay out (security).
- A test checks that the file matches the image tag in `IMAGES.lock`. Bumping the image means regenerating the file, and the
  diff shows exactly which knobs changed.

## 4. Open questions
1. backend: does `expand()` belong in `graph/` (my proposal) or in the app before `validate()`?
2. design/frontend: is `inner_path` on Issue enough to highlight inside a collapsed group?
3. security: may user groups embedded in a shared workflow carry params defaults (treated as untrusted, validated like any node)?
