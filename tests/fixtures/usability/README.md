# Usability-test fixtures (design/research/usability-test-plan.md)

Owner: bao-tester. Used by the S-5 slice with `FakeWorker` (no GPU). Each file maps to a task:

| File | Task | Expected (see `expected_issues.json`) |
|---|---|---|
| `t1_lora_template_smoke.vlmflow.json` | T1/T2 | validates clean; node `data` is the "👉 replace with your data" node |
| `t3_wrong_wire_missing_input.vlmflow.json` | T3 | CFG-001 (Dataset → Model port on `eval`) + CFG-003 (`train.base_model` unconnected) |
| `t4_workflow.vlmflow.json` + `t4_mem001_preflight.json` | T4 | MEM-001 `{need_gb: 48, avail_gb: 42}` injected by an estimator stub |
| `t5_shared_unverified_gpu_node.vlmflow.json` + `t5_app_config.json` (policy_block off) / `t5b_app_config.json` (on) | T5 / T5b | CFG-004 not_installed / unverified, `suggested_package=null`, hint string never shown |

Assumptions to re-check when the real node registry lands (S-2/S-5): port names (`dataset`, `model`,
`base_model`, `adapter`), param names (`run_mode`, `tier`), and the placeholder revision `"0"*40`.
`expected_issues.json` doubles as the oracle for an automated check, so the session itself is not the test.

Safety: `policy_block_gpu_third_party: false` is accepted only with `worker: fake` (spec in tests/unit/security).
