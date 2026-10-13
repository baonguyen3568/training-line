# scripts/: index of manual tools

Scripts that nothing else in the repo calls (no CI gate, test or other script). The owner wrote each line. Every
such script has an entry here; one without an entry is a candidate for deletion (M3 D-6, bao-cleancode).
Gates and entrypoints wired into `scripts/check.sh` or the app are documented where they are called.

| Script | Owner | What it does / how to run |
|---|---|---|
| `scripts/fixcode/meminfo_sampler.sh` | bao-fixcode | Logs host MemAvailable/Cached/SwapFree + the GPU-lock flag every 60 s (default 48 h) to `outputs/system/meminfo-<ts>.jsonl` (schema `vlm.system.meminfo/v1`): data to recalibrate the memguard thresholds (ADR-005 D8) and the dashboard RAM history. Read-only. `nohup setsid bash scripts/fixcode/meminfo_sampler.sh [interval_s] [duration_h] &` |
| `scripts/probe/parse_build_check.py` | bao-fixcode | P-0: builds Nemotron-Parse-v1.1 from the prepared config only (meta device, offline, empty HF modules cache) and proves the RADIO code comes from the local pinned copy, not the Hub; prints native/remote class. Run inside `vlm-automodel` with `--network none`. |
| `scripts/probe/parse_generate_timing.py` | bao-fixcode | BUG-001/TD-1: Parse generation wall-clock vs token count with the native AutoModel class (patch 0003, no KV cache) on NVIDIA's golden image; numbers in `third_party/patches/automodel/VENDORED.md` §0003. Inside `vlm-automodel` on the GPU (GPU lock). |
| `scripts/qa/mem_summary.py` | bao-tester | Summarizes the MemAvailable/peak CSV written by `scripts/qa/watch_container_mem.sh` into baseline / peak / steady numbers for a perf record (skill `tester-perf-bench` §1). |
| `scripts/qa/parse_native_remote_parity.py` | bao-tester | Parity gate for a Parse checkpoint: native vs remote class (strict load vs the base, fp32 log-probs atol 1e-3, 001a) before trusting F1 from the remote eval path. Inside `vlm-automodel`, GPU lock. |
| `scripts/qa/probe_teacher_forced.py` | bao-tester | Teacher-forced loss / token accuracy of a Parse checkpoint on N images (per-checkpoint gate P-3b, 8aac26a). Inside `vlm-automodel`, GPU lock. |
| `scripts/probe/engine_schema_write.py` | bao-cleancode | P016 §3: probes the pinned train image offline (`engine_schema.py` inside it), filters through `nodes/advanced_denylist.py` and writes `configs/engine/automodel-<tag>.json` (the Advanced params). Rerun after an image bump; the diff shows which knobs changed. `uv run python scripts/probe/engine_schema_write.py` |
| `scripts/dev/run_workflow.py` | bao-backend | Runs one workflow file through the app's executor without HTTP (dev / E2E of the real worker). |
| `scripts/dev/seed_runs.py` | bao-backend | Writes fake run dirs covering all 8 `RunStatus` values, for frontend and tester work. |
| `scripts/probe/gpu_probe.py` | bao-backend | GPU environment probe inside the NGC container, no training (G0 / S-2). |
| `scripts/team/start-team.sh` | PM | Starts the team agents in tmux sessions `bao-trainning-2..10` (idempotent). |
| `scripts/team/stop-team.sh` | PM | Stops tmux sessions `bao-trainning-2..10`; never touches `bao-trainning-1` (PM). |
