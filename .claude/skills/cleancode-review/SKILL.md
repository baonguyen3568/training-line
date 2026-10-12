---
name: cleancode-review
description: Architecture/clean-code review of a diff, PR, branch or file set in the VLM pipeline repo. Use when asked to review code, before approving a merge, or when checking module boundaries, abstraction level, duplication and typing in src/vlm_pipeline.
---

# Clean-code review (bao-cleancode)

Goal: readable, thin, layered code. Evidence-based comments; each finding = file:line + why + concrete fix.

## 1. Collect the change
```bash
git status --short
git diff --stat main...HEAD    # or: git diff --stat / git diff <base>
git diff main...HEAD -- src tests configs
```

## 2. Run machine gates first (do not hand-review what tools catch)
Use the `cleancode-check` skill (`scripts/check.sh` once it exists). Record failures, don't re-explain them.

## 3. Human checklist (severity: BLOCKER / MAJOR / NIT)
- **Layering** (BLOCKER): stages `data|train|quantize|eval|serve` must not import each other; all may import `common`; only CLI wires stages. Stages talk via artifacts on disk + manifest JSON.
- **Heavy imports** (MAJOR): `torch`, `transformers`, `nemo*`, `megatron*`, `modelopt`, `tensorrt_llm`, `vllm` only inside functions or adapter modules, never at top level of CLI/config modules.
- **Abstraction level** (MAJOR): no base class/protocol/factory/registry with a single implementation; rule of three. Thin wrapper over NVIDIA libs, don't re-implement what the recipe already provides.
- **Duplication** (MAJOR): search before accepting new helpers: `grep -rn "def <name>" src/`.
- **Config** (MAJOR): pydantic v2, `extra="forbid"`, no hard-coded paths/hyperparams; resolved config + git SHA + lib versions written to output dir.
- **Typing** (MAJOR for public API): full annotations; `Any` justified; `# type: ignore[code]` with code only.
- **Errors** (MAJOR): no bare/blanket except that swallows; context in messages; fail fast on bad config.
- **Tests** (MAJOR): pure logic unit-tested; GPU paths marked `gpu`/`slow`; no real model download in default test run.
- **Safety** (BLOCKER): no secrets, no writes outside repo, heavy jobs use `.team/GPU_LOCK`.
- **Readability** (NIT): names, function length (>50 lines → question it), Google-style docstrings that say *why*.
- **Diff hygiene** (NIT): one purpose; refactor separate from behavior change.

## 4. Report
Write findings grouped by severity. Send to author via SendMessage (summary + top BLOCKER/MAJOR), log one line in `.team/log/YYYY-MM-DD.md`:
`HH:MM [bao-cleancode → <author>] review <PR/branch>: N blocker, M major`.
If fixing is delegated, tell bao-fixcode which files so nobody edits the same file concurrently.
Disagreement not resolved after one exchange → `.team/CONFLICTS.md` + notify PM (`trainning-model-6a`).
