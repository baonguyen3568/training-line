---
name: frontend-benchmark-viz
description: Use when building or reviewing any benchmark comparison view (base vs fine-tuned vs quantized, accuracy and latency/throughput) — rules for tables, deltas, trade-off plots and reproducibility metadata so comparisons are not misleading.
---
# Benchmark comparison views — rules

## Main table
- Rows = benchmark (+ metric), columns = variants ordered: published → base → fine-tuned → quantized (bf16 → fp8 → nvfp4 → int4).
- Cell = value (fixed decimals per metric) + delta vs base (sign-colored, with "+"/"−"), stderr/CI if available, small n.
- Mark `source=published` visually distinct (italic/hatched); never average measured with published.
- Gray-out / warn when cells in a row differ in `harness_version`, `prompt_hash` or `n_samples`.

## Charts
- Accuracy vs throughput scatter (one point per variant, label it) = the key "which quantized model to ship" view.
- Grouped bars / dot plot with a baseline line for per-benchmark deltas. No radar charts for decisions.
- Perf always shows its conditions: batch size, input resolution / frames, output length, TTFT p50/p95.
- Use bao-design color tokens; variant colors stable across all pages.

## Reproducibility footer (every view)
harness + version, git commit, config hash, date, hardware ("1x GB10, shared host"), whether GPU lock was held.

## Review checklist
- [ ] Can a reader tell in 5s which variant is best for accuracy and for speed?
- [ ] Deltas computed against the correct base (same harness/prompt)?
- [ ] Missing values show "—" with tooltip reason, not 0.
- [ ] Sorting/filtering does not hide the base column.

## Added after review (bao-design, bao-tester)
- Never share an axis between metrics on different native scales; stored `value` is 0-100 (see contract `scale`).
- If |delta| is smaller than its paired CI (`delta_ci95_low..high` spans 0) → label "within noise", no win/loss color.
- Do NOT infer "no difference" from overlapping per-variant error bars; only `delta_ci95_*` decides.
- Status color always paired with an icon/text (sanity gate PASS/FAIL, "not reproducible").
- Colors: hue = lineage (base / fine-tuned / ft#2), precision = lightness + marker (BF16 ● FP8 ▲ NVFP4 ◆ INT4 ■); tokens `color.variant.<lineage>.<precision>` from bao-design (design-benchmark-viz skill owns color/form).
