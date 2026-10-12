---
name: tester-bug-report
description: Use when a test fails, an eval/benchmark looks wrong, or a pipeline step crashes and the issue must be handed to bao-fixcode — produces a minimal reproducible bug report, logs it, and defines how the fix is verified (regression test). Also use when confirming/closing a fix.
---

# Bug report & fix verification loop (tester ⇄ fixcode)

## 1. Before reporting
- Re-run once to rule out flakiness (note if intermittent + frequency).
- Shrink: smallest model (`hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration` if the bug is not
  model-specific), smallest data (1–8 samples), CPU if possible. Remove unrelated flags.
- Capture env: `bash .claude/skills/fixcode-triage/scripts/env_fingerprint.sh` (preferred — MemAvailable/SwapFree, versions, GPU_LOCK, no secrets); else `git rev-parse --short HEAD`, `uv pip freeze | egrep 'torch|transformers|vllm|modelopt|nemo|lmms|vlmeval'`,
  container image+digest, `python -c "import torch;print(torch.__version__, torch.version.cuda)"`.

## 2. Report format — file `.team/bugs/BUG-NNN-<slug>.md`
```
# BUG-NNN: <one-line symptom>
Severity: blocker | major | minor     Found by: bao-tester   Date: YYYY-MM-DD
Component: data | train | quantize | eval | serve | common
## Repro (copy-paste)
<exact commands, ≤ 10 lines>
## Expected
## Actual  (error tail / wrong number, ≤ 30 lines; full log path)
## Env  (paste env_fingerprint.sh output)
## Memory/hang bugs: MemAvailable at start + peak; did the host freeze? (UMA may freeze instead of OOM)
## Notes / suspected cause (optional, clearly marked as guess)
## Verification  (test that must pass after fix: tests/...::test_...)
Status: OPEN | FIXING | FIXED-UNVERIFIED | VERIFIED | WONTFIX
```

- NEVER reproduce a heavy OOM on the shared host before messaging bao-fixcode.

## 3. Hand-off
- SendMessage to `bao-fixcode`: first line = "BUG-NNN [severity]: <symptom>", then path to the report.
- Append to `.team/log/<date>.md`: `HH:MM [bao-tester → bao-fixcode] BUG-NNN ...`.
- Blocker ⇒ also notify `trainning-model-6a` (bao-pm).

## 4. Verify the fix
- Write/keep a failing regression test FIRST (in `tests/`, marked appropriately), confirm it fails on the old code.
- After fix: run the regression test + the related test module + `pytest -m "not gpu and not slow"`.
- Pass ⇒ Status VERIFIED, reply to bao-fixcode, log line. Fail ⇒ back to FIXING with new evidence.
- Fixes touching eval/metrics ⇒ also re-run the tester-eval-sanity smoke (`--limit 20`) on base model.
