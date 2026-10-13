# Usability-test fixtures (design/research/usability-test-plan.md)

Owner: bao-tester. Used by the S-5 slice with `FakeWorker` (no GPU). Each file maps to a task:

| File | Task | Expected (see `expected_issues.json`) |
|---|---|---|
| `t1_invoice_to_json_smoke.vlmflow.json` | T1/T2 | template `invoice_to_json` (Nemotron-Parse v1.1, full fine-tune), validates clean (verbatim template workflow, node ids renamed model/data/split/train/eval); node `data` is the "👉 replace with your data" node |
| `t3_wrong_wire_missing_input.vlmflow.json` | T3 | CFG-001 (`split.val` Dataset → `eval.model`) + CFG-003 (`train.base_model` unconnected) |
| `t6_parse_on_lora.vlmflow.json` | T6 (optional) | CFG-010 on `train`: Nemotron-Parse wired into LoRA; text names the model and says "fine-tune toàn phần", no raw node type ids |
| `t4_workflow.vlmflow.json` + `t4_mem001_preflight.json` | T4 | MEM-001 `{need_gb: 14, avail_gb: 9}` (illustrative until P-1), `already_optimized: true`, actions = queue_until_memory + see_memory_usage |
| `t5_shared_unverified_gpu_node.vlmflow.json` + `t5_app_config.json` (policy_block off) / `t5b_app_config.json` (on) | T5 / T5b | CFG-004 not_installed / unverified, `suggested_package=null`, hint string never shown |

Fixtures are derived from `configs/templates/invoice_to_json.json` against the real node registry (2026-10-13).
`expected_issues.json` is the oracle of `tests/unit/graph/test_usability_fixtures.py` (exact codes, node ids, ports,
key params, and the rendered vi text for `must_not_contain`), so the session itself is not the test. Preflight issues
(`via` a stub, e.g. MEM-001) are not checked there: those workflows must validate clean.

Safety: `policy_block_gpu_third_party: false` is accepted only with `worker: fake` (spec in tests/unit/security).
