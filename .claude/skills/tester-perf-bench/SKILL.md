---
name: tester-perf-bench
description: Use when measuring latency, throughput or peak memory of a VLM (before/after quantization, serving config changes, or regression checks) on the shared DGX Spark GB10 with unified memory — gives the measurement protocol, commands and the pitfalls (nvidia-smi shows no memory on GB10, shared host noise, warmup/JIT spikes).
---

# Performance benchmark protocol — DGX Spark GB10 (UMA, shared host)

## 0. Pre-flight
- GPU lock: create `.team/GPU_LOCK` (owner, job, start, PID); abort if it exists.
- Record host state: `free -g`, `cat /proc/meminfo | egrep 'MemTotal|MemAvailable|Cached'`,
  `docker ps --format '{{.Names}}'` (other services = noise), `nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm --format=csv`.
- Do NOT run `drop_caches` (needs sudo — forbidden). Record `Cached` instead.
- Check MemAvailable leaves headroom: model + KV budget must be < MemAvailable − 10GB, otherwise stop and tell bao-pm.

## 1. Memory measurement (nvidia-smi Memory-Usage = "Not Supported" on GB10 — expected)
- Host-level sampler (run in background, 1 Hz, to CSV):
```bash
while true; do awk -v ts="$(date +%s)" '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{printf "%s,%.2f\n", ts,(t-a)/1048576}' /proc/meminfo; sleep 1; done > outputs/perf/mem_$(date +%s).csv
```
  Report: baseline (before start), startup peak (load + JIT/compile), steady state after warmup, peak during load test.
  Model footprint ≈ steady − baseline.
- In-process (PyTorch): `torch.cuda.reset_peak_memory_stats()` … `torch.cuda.max_memory_allocated()/2**30`.
- Report both; they differ (allocator vs system incl. caches, JIT, CUDA context).

## 2. Serving benchmark (vLLM / TRT-LLM OpenAI endpoint)
- Fix: `--gpu-memory-utilization` (same value for every variant, ≤0.5 on this shared host), eager vs cuda-graph
  (record it), max_model_len, image size (e.g. 448² and 1024²), input text len, output len (e.g. 128).
- Warmup ≥ 10 requests AND past JIT compile (~25s on sm_121, see .team/knowledge/fixcode.md §2.3), discarded.
- Always benchmark both FP8 and NVFP4 — NVFP4 can be ~25-30% SLOWER than FP8 on GB10 (no native sm_121a FP4 kernel → Marlin fallback).
- Memory budget: `torch.cuda.mem_get_info()` under-reports on UMA — use MemAvailable.
- vLLM: `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model <m> --dataset-name random-mm ... --num-prompts 200 --max-concurrency {1,4,16} --save-result --result-dir benchmarks/perf/`
  (check `vllm bench serve --help` for the exact random-mm flags of the pinned version).
- TRT-LLM: `benchmark_serving.py` random image dataset (see TRT-LLM docs for the pinned version).
- Metrics: TTFT, TPOT/ITL, E2E latency p50/p90/p99; request/s; output tok/s; peak memory.
- Repeat ≥3 times, report median and spread. Log actual kernel backend (e.g. FP4 CUTLASS vs Marlin fallback on sm_121).

## 3. Offline / in-process microbench
- `torch.cuda.synchronize()` around timed regions; use `time.perf_counter()`; exclude first N iterations.
- Separate: image preprocessing, vision encoder, prefill, decode.

## 4. Result format: `benchmarks/perf/<model>-<precision>-<date>.json`
Must include: git sha, container digest, engine+version, all flags, host baseline mem, other containers running,
GPU temp/clock, repeats, per-run raw numbers.

## 5. Comparison before/after quantize
Same engine, same flags, same images/prompts, same concurrency, run back-to-back. Pair with accuracy delta
(tester-eval-sanity) — a speedup with unacceptable accuracy drop is a FAIL.

## 6. Cleanup
Stop servers (only your own `vlm-*` containers / your PIDs), kill memory sampler, `rm -f .team/GPU_LOCK`.
