---
name: design-dashboard-review
description: Use when designing, reviewing or giving UX feedback on any dashboard screen/component of this project (run list, run monitor, run compare, model scorecard, quantize/benchmark view, export, demo inference) or on Proposal 006 — user-flow mapping, DGX Spark/GB10-specific widgets (unified memory, GPU_LOCK), 4-state check, design-token compliance.
---

# Dashboard UX review (bao-design)

Background: `.team/knowledge/design.md` (§1 flow, §3 GB10, §9 checklist). Tokens: `design/tokens/` (once created).

## 1. Map the screen to the flow
Main flow: **create run → monitor → compare → pick champion → quantize → benchmark → export**.
For the screen under review write one line: `Step <n> — primary question: "<...>"`.
If it answers no step's question, or more than one primary question → push back (split or cut).

## 2. Global chrome (must exist on every page)
- GPU_LOCK banner: read `.team/GPU_LOCK` → "Busy: <who> — <job> — since <time> — PID". Create-run disabled with the reason.
  **Never** a kill/stop button for someone else's job (CLAUDE.md safety rule).
- Unified memory bar (128GB, stacked): other services | vlm-* jobs | buff/cache | free.
  Source = `/proc/meminfo` (MemTotal − MemAvailable), NOT nvidia-smi (reports "Not Supported" on GB10).
  Status thresholds come from bao-cleanup/bao-backend — do not invent numbers.
- Theme toggle (light/dark), honoring `prefers-color-scheme`.

## 3. Per-panel checklist
- [ ] 4 states designed: empty · loading · error · stale (data older than N min, show timestamp)
- [ ] Units shown on every number (tokens/s vs samples/s vs images/s; ms p50/p95; GB)
- [ ] Tables: sortable, clickable rows, tabular-nums, tooltip only when truncated, run_id in mono
- [ ] Run selection survives Table ⇄ Chart toggle; baseline can be pinned; config compare has "Diff only"
- [ ] Color follows entity (stable per run_id), never rank; ≤8 categorical; status = color + icon + label
- [ ] Live refresh 5–10s with pause; no 1s polling (shared host)
- [ ] Every benchmark number links to lineage (dataset → ckpt → quantized → engine → harness version/seed/n)
- [ ] VLM error analysis view: image/video frame + prompt + ground truth + prediction side by side
- [ ] Keyboard reachable, focus visible, text contrast ≥4.5:1, non-text ≥3:1 in BOTH themes
- [ ] Charts pass `design-benchmark-viz` skill checks

## 4. Output format
Write review as:
```
## Review: <screen> (<date>) — bao-design
Verdict: APPROVE | CHANGES REQUESTED
Blocking: 1. ... (why, evidence/source)
Non-blocking: 1. ...
Questions for <owner>: ...
```
Append to the PR/proposal discussion; log 1 line in `.team/log/YYYY-MM-DD.md`.
Disagreement not resolved after one round → `.team/CONFLICTS.md` + message PM (`trainning-model-6a`).

## 5. Low-fi wireframe convention
ASCII wireframes in proposals (fit 100 cols), e.g.
```
┌ GPU: BUSY bao-backend sft-lora 14:02 PID 4121 ┐ ┌ RAM ███████▒▒░░ 92/128GB ┐
│ Runs [table|chart]  filter: tag:champion  [+ New run (disabled)]              │
```
Hi-fi only after Proposal 006 is accepted (ADR).
