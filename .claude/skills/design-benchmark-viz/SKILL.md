---
name: design-benchmark-viz
description: Use when choosing or reviewing a chart for training metrics or benchmark results in this project — base vs fine-tuned vs quantized comparisons (MMMU, MMBench, DocVQA, ChartQA, OCRBench, MathVista, Video-MME…), accuracy-vs-latency/memory Pareto plots for FP8/NVFP4/INT4-AWQ, loss curves, stat tiles, or the project's chart color palette.
---

# Benchmark & metrics visualization (bao-design)

Also load the built-in `dataviz` skill for mark specs and the palette validator. Knowledge: `.team/knowledge/design.md` §4–§5.

## 1. Pick the form by the question
| Question | Form | Never |
|---|---|---|
| A vs B across N benchmarks | Table with Δ column + dot plot / grouped bar; small multiples per benchmark | Radar chart |
| What does quantization cost/buy? | Scatter: x = latency p50 ms (or peak unified-mem GB), y = accuracy; Pareto front line; **quality floor** line; best corner top-left | Dual y-axis |
| Δ vs BF16 baseline | Diverging bar around 0 (blue/red, gray mid) | Absolute numbers without baseline |
| Loss / lr / grad-norm over steps | Line 2px, raw faint + smoothed, NaN/inf markers | Smoothing that hides raw |
| One headline number | Stat tile: value + unit + Δ vs baseline | 1-bar bar chart |

Different scales (OCRBench 0–1000 vs % benchmarks) never share an axis — normalize with label, or facet.

## 2. Mandatory metadata on every benchmark chart (caption or footer)
Hardware (`DGX Spark GB10`), engine + version (TRT-LLM/vLLM/HF), precision, batch, max seq len,
image resolution / video frames, harness + version (VLMEvalKit / lmms-eval / NeMo Evaluator), n samples, seed, date.
Missing any → show badge "not reproducible" and flag to bao-tester.
Differences smaller than the CI / noise band → label "within noise", do not crown a winner.
Ask bao-research for metric direction (higher better?), units, and whether a judge model is used.

## 3. Color assignment (fixed project convention — tokens in `design/tokens/color.{light,dark}.tokens.json`)
**Hue = model lineage, precision = lightness step (away from surface) + marker shape.**
| lineage \ precision | BF16 ● | FP8 ▲ | NVFP4 ◆ | INT4-AWQ ■ |
|---|---|---|---|---|
| base light / dark | `#2a78d6` / `#3987e5` | `#2164b4` / `#5b9ef2` | `#195193` / `#7bb5fe` | `#113e74` / `#9acbff` |
| ft light / dark | `#eb6834` / `#d95926` | `#ca5a2e` / `#e9774e` | `#aa4c28` / `#f89370` | `#8b3f21` / `#ffaf91` |
| ft2 light / dark | `#1baf7a` / `#199e70` | `#1b9669` / `#4eb288` | `#1b7e58` / `#74c5a0` | `#196748` / `#96d9b9` |
- `published` (model card) = hollow 2px ring in the model's hue + italic label; never averaged with measured.
- Scatter/Pareto: shape + direct label on every point are MANDATORY (cross-lineage deep steps fail CVD all-pairs); max 2 lineages.
- Many SFT runs in one chart (not variants) → categorical `color.series.1..8`, fixed per run_id, never recolor on filter.
- Δ vs baseline: `color.delta.better|worse` + ▲/▼ only when delta_ci95 excludes 0; else `text.muted` "no difference".
  Never infer significance from overlapping error bars (use paired delta_ci95 from eval).
- Status (`good #0ca30c`, `warning #fab219`, `serious #ec835a`, `critical #d03b3b`) only for state, always + icon + text.
- NVIDIA green `#76B900` is chrome accent only, never a series.

## 4. Validate before shipping
```bash
node <dataviz-skill-dir>/scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
node <dataviz-skill-dir>/scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark --pairs all
# each precision ramp: add --ordinal, e.g. "#2a78d6,#2164b4,#195193,#113e74" --mode light --ordinal
```
Then render and eyeball: label collisions, legend present (≥2 series), table view available, both themes.

## 5. Static reports (matplotlib/plotly from backend)
Same tokens; export PNG at 2x + SVG; light theme for docs, both if embedded in dashboard.
Font: system sans; numbers in tables tabular. Save to `benchmarks/<run_id>/figures/`.
