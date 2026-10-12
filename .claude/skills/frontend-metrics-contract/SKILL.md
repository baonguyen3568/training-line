---
name: frontend-metrics-contract
description: Use when defining, changing or consuming the run/metrics/benchmark data contract between pipeline (backend) and dashboard — file layout, JSON/JSONL fields, schema_version rules, and how to validate a run directory.
---
# Metrics & benchmark data contract (draft — final version lives in Proposal 006 / ADR)

## Layout (file-based, local-first; dashboard is a read-only consumer)
```
outputs/runs/<run_id>/       # merged layout (bao-backend + bao-cleanup + bao-frontend), pending Proposal 005/006
  run.json         # metadata, written at start, updated at end (status)
  metrics.jsonl    # 1 line per log step (train/eval scalars)
  system.jsonl     # 1 line per ~5-10s (host memory, GPU util/clock/power/temp, job RSS)
  events.jsonl     # checkpoint saved, eval started, error, finished
  logs/ checkpoints/ export/   # retention may prune checkpoints; the 4 files above + logs/ are whitelisted (never deleted)
outputs/system/disk_report.json   # by bao-cleanup (P005 D6): schema_version, ts, free_gb, mem_available_gb, swap_used_gb, ..., level(ok|warn|critical), reasons[{metric,value,threshold,level}]
benchmarks/<model_variant>/<benchmark>.json   # flat records, one per (variant, benchmark, metric)
```

## Other sources
- `outputs/system/meminfo-<ts>.jsonl` (bao-fixcode sampler, schema `vlm.system.meminfo/v1`, 60s): `schema_version, ts, mem_total_gb, mem_available_gb, cached_gb, swap_free_gb, gpu_lock_held` → history for memory widget (use mem_available_gb, never GPU memory).
- Error records (P007 R9, schema `vlm.error/v0`, samples `.team/knowledge/error-samples-v0.jsonl`; code format `<GROUP>-<NNN>` e.g. MEM-003, `slug` for engineers only; copy from i18n `error.<code>.title|what|why` — `*_vi` in samples are fixture-only): `{schema_version, ts, phase(validate|runtime), code, slug, severity(error|warn|info), node_id, port?, params{}, why_vi, actions[{label_vi, kind(patch|resume|navigate), target, patch?:[{node_id,param,value}], confirm(bool)}], technical_detail}`; same `code` as `/api/graphs/validate` for preflight errors. Status `interrupted` (exit 75) → "resume from checkpoint".

## Events used by UI
- `checkpoint_pruned` (bao-cleanup, P005 D7 §7): `{event, path, action(strip_optim|delete), bytes, ts, by, approved_by}` → UI shows "optimizer stripped" vs "deleted at <ts> by <by>".
- `cache_evicted` (P009 §5 / P005 §6 R1): `{event, run_id, cache_key, bytes, ts, by}`.

## Records (every file/line carries `schema_version`)
- run.json: `schema_version, run_id, name, task(sft|lora|grpo|dpo|quantize|eval), base_model, dataset, config_path, config_hash, git_commit, seed, started_at, ended_at, status(running|finished|failed|killed|interrupted|evicted — evicted: heavy bytes freed, history kept → UI "đã giải phóng, chạy lại để tính lại"), pid, tags[], stage, parent_run_id (quantize/eval → train run, builds lineage base→sft→fp8/nvfp4), container_image, container_digest`
- metrics.jsonl: `{"schema_version":1,"ts":<unix float>,"step":int,"epoch":float,"split":"train|val","loss":..,"lr":..,"grad_norm":..,"tokens_per_s":..,"samples_per_s":..,"mem_max_alloc_gb":..}` — extra keys allowed (flat, numeric), never nested.
- system.jsonl: `ts, host_mem_total_gb, host_mem_avail_gb, gpu_util_pct, gpu_clock_mhz, gpu_power_w, gpu_temp_c, job_rss_gb (container cgroup memory.current), job_gpu_mem_gb (per-process from nvidia-smi --query-compute-apps)` — null when unsupported. On UMA CUDA allocations may not be in cgroup → UI "vlm jobs" = job_rss_gb + job_gpu_mem_gb, labelled "estimate".
- benchmark record (flat, agreed with bao-research for Proposal 002): `schema_version, model_id, variant(base|sft|lora|...), precision(bf16|fp8|nvfp4|int4_awq), benchmark, split, metric, value, scale, higher_is_better(bool — drives Δ color direction), n_samples, stderr, ci95_low, ci95_high, sanity_gate(pass|fail|null — base-model reproduction gate from bao-tester), source(measured|published), harness, harness_version, prompt_hash, config_hash, git_commit, date, thinking(bool), judge_model(null if rule-scored), max_new_tokens, seed, dataset_revision, container_digest, chat_template_hash, image_cfg_hash (or min_pixels/max_pixels), vision_precision, eval_path(endpoint|inprocess), samples_path (raw outputs for audit), answer_extractor(rule|judge), kernel_backend, repeat_spread (max−min over repeated runs of same config, null if not measured), repeat_n, unparseable_rate`
  - Benchmark metadata (scale, higher_is_better, judge, split, display name) lives in a data **registry** (configs/eval, owned by bao-research), never hard-coded in UI.
  - Paired comparison: optional `compare_to` (record id / variant of baseline), `delta`, `delta_ci95_low`, `delta_ci95_high` — paired CI computed by eval, NOT inferred from overlapping error bars in the UI.
  - UI badge "not reproducible" when any of git_commit, container_digest, config_hash, dataset_revision, harness_version, seed is missing.
  - `value` always on 0-100 scale (OCRBench /1000 → /10, ANLS ×100); `scale` documents it.
  - Perf = separate flat records with `benchmark="perf"`, `metric` in {ttft_ms_p50|p90|p99, tpot_ms_p50|p90|p99, e2e_ms_p50|p90|p99, tokens_per_s, requests_per_s} (each may carry `_min`/`_max` across iterations), plus `memory_startup_peak_gb, memory_steady_gb, memory_peak_load_gb` (from /proc/meminfo), `torch_max_alloc_gb`, and condition keys (= `perf_conditions`, from tester-perf-bench): `engine, engine_version, kernel_backend (e.g. FP4 CUTLASS vs Marlin fallback), precision, vision_precision, gpu_memory_utilization, cuda_graph(bool), image_size, num_frames, input_tokens, output_tokens, concurrency, iterations, warmup, host_baseline_used_gb, other_containers[]`.
  - UI: compare two perf records only if `perf_conditions` match; otherwise show a diff-of-conditions warning.
  - UI: error bars from ci95_*, warning badge on any number whose base model has `sanity_gate=fail`.

## Pending (asked bao-backend 2026-10-12)
- `samples_path` → `predictions.jsonl` per (variant, benchmark): `sample_id, image_ref (relative to data root), question, gt, pred, correct(bool), score, extractor(exact|regex|llm-judge)` — backend writes adapters for lmms-eval (--log_samples) and VLMEvalKit (xlsx/tsv) — needed for Error analysis (P1).
- `.team/GPU_LOCK` has `host_pid` (orchestrator PID on host, held for the whole job) — agreed in P008.

## Rules
1. Additive change (new optional key) → no version bump. Rename/remove/semantics change → bump `schema_version`, dashboard supports N and N-1.
2. Units in the key name (`_gb`, `_ms`, `_pct`, `_w`, `_c`). Timestamps = unix seconds UTC.
3. Single pydantic definition in `src/vlm_pipeline/common/` — backend writes with it, dashboard reads with it.
4. Writers append + flush per line (so tailing is safe); no rewriting metrics.jsonl.

## Validate a run dir
```bash
uv run python -c "
import json,sys,pathlib
d=pathlib.Path(sys.argv[1]); json.loads((d/'run.json').read_text())
for f in ['metrics.jsonl','system.jsonl','events.jsonl']:
    p=d/f
    if p.exists():
        bad=[i for i,l in enumerate(p.read_text().splitlines()) if l.strip() and 'schema_version' not in json.loads(l)]
        print(f, 'OK' if not bad else f'missing schema_version at lines {bad[:5]}')
" outputs/runs/<run_id>
```
(Replace with the pydantic-based validator CLI once it exists.)

## Change process
Propose change → message bao-backend (+ bao-research for benchmark fields) → update Proposal/ADR → then code.
