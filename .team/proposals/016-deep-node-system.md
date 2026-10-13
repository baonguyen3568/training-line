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
  (`node_id`) + `inner_path` = **array of ids per level** (e.g. `["train", "loop"]`, design), keeping the inner node's
  `port`/`param`; errors of the group itself (CFG-009, bad definition) have no `inner_path`. Provenance records the hash
  of each group definition used.
- Limits (security), enforced **while** expanding: depth ≤ 3, no recursion, ≤ 500 nodes after expansion, ≤ 50 group
  definitions per file, existing file-size limit. Group versions pinned like node versions (CFG-009).
- **User groups are untrusted data** (security): ids only in the `user.<slug>` namespace, and a clash with a node type
  or first-party group is an error (a shared file can never redefine `train.lora_vlm`). Plain-text title/description/
  exposed names (length-capped, rendered as untrusted text, ASCII ids + lookalike check), no `title_key`. Every
  `inputs/outputs/params` mapping must point at a real inner node/port/param. Trust is per inner node type (D16): a group
  never raises it, and a collapsed group containing an unverified/plugin node shows that badge. Param defaults are
  allowed (validated after expansion like any node).
- `available=False` nodes: greyed with a lock in the library, can be placed, validation reports CFG-011 (design).
- The "big" beginner nodes (`train.lora_vlm`…) can become first-party groups later **without changing saved workflows**:
  the type id stays and resolves to a group.

## 3. Advanced params generated from the engine: everything valid, minus a denylist by rule
- A script run inside the pinned engine image (`scripts/probe/engine_schema.py`) walks the engine's own config schema
  (recipe dataclasses / defaults of the pinned version) and dumps **every** option it finds (dotted key, type, default,
  bounds, doc) to `configs/engine/<engine>-<image tag>.json`, committed. Coverage target (PM, user "cực sâu"): nearly the
  whole engine surface, not a hand-picked list.
- Excluded by rule (security owns it): one denylist module `nodes/advanced_denylist.py` imported by the generator (refuses
  to emit) **and** by a test over the COMMITTED file (the gate: the file can be hand-edited). Key segments, case-insensitive:
  `_target_ _partial_ _args_ _recursive_ _convert_ trust_remote_code` and path/dir/file/url/uri/module/class/cls/fn/func/
  callable/import/hook/plugin/script/cmd/command/env/token/key/secret/password/revision/repo/cache/endpoint/host/port.
  Node-controlled keys (dirs, batch, lr…) stay node params.
- Each node's `params_model` gets `advanced: AdvancedOverrides` built from that file: `extra="forbid"`, types bool/int/
  float/enum/short scalar lists, any remaining str field with a strict pattern, **any value containing `${` or `oc.` is
  refused** (OmegaConf resolvers such as `${oc.env:HF_TOKEN}` would leak secrets; tested). `x-ui.advanced=true`.
- A test ties the file to the image tag in `IMAGES.lock`; an image bump regenerates it and the diff shows which knobs changed.

## 4. Open questions
1. backend: does `expand()` belong in `graph/` (my proposal) or in the app before `validate()`?
2. ~~design: is `inner_path` enough?~~ Yes (array + inner port/param); display spec goes to P4-4 (design).
3. ~~security: user group param defaults?~~ Yes, validated after expansion; rules folded into §2/§3 above.
