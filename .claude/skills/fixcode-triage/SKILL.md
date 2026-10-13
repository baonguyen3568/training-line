---
name: fixcode-triage
description: Use when picking up a bug (BUG-NNN from bao-tester/security/cleancode) or any crash/wrong result in the VLM pipeline — reproduce, fingerprint env, classify (ABI/sm_121, UMA memory, data/processor, checkpoint, numerics, concurrency), find root cause, apply a minimal fix with a regression test, and close the loop with the reporter.
---

# Bug triage → root cause → minimal fix (bao-fixcode)

Knowledge base with known GB10/aarch64 failure modes: `.team/knowledge/fixcode.md` (read §2 for the class you suspect).

## 1. Intake
- Read `.team/bugs/BUG-NNN-*.md`. Set `Status: FIXING`, put `bao-fixcode` as owner.
- Tell bao-cleancode which files you are holding: SendMessage `bao-cleancode` → "đang giữ: <paths> (BUG-NNN)". Release when done.
- Blocker → notify PM `trainning-model-6a`.
- Create `outputs/debug/BUG-NNN/` and keep `outputs/debug/BUG-NNN/STATUS` in sync (one line: `OPEN|FIXING|FIXED|VERIFIED YYYY-MM-DD`) — bao-cleanup's retention reads it (delete allowed VERIFIED + 14 days).
- `[SEC]` bugs (from bao-security): never write secret values into bug files/logs, only locations; bao-security verifies before close. Any wheel outside PyPI/NGC → send bao-security URL + sha256 + source commit + reason BEFORE installing.

## 2. Reproduce (no repro = no fix)
- Run the repro exactly. Then shrink: tiny model (`hf-internal-testing/tiny-random-*`), 1–8 samples, CPU if bug is not GPU-specific.
- Capture fingerprint: `bash .claude/skills/fixcode-triage/scripts/env_fingerprint.sh > outputs/debug/BUG-NNN/env.txt`
- GPU repro → never by hand: run it as a job (`uv run python -m vlm_pipeline.common.jobrun_cli --who bao-fixcode --tag
  bug-NNN --est-peak-gb N … -- <cmd>`), which runs preflight, waits for memory, takes and releases the GPU lock.
  Status only: `uv run python -m vlm_pipeline.common.gpu_lock status`. Never write or delete `.team/lock/GPU_LOCK`
  yourself (a hand-written lock counts as corrupt; CLAUDE.md).
- Memory-heavy repro → follow skill `fixcode-uma-oom` (risk of freezing the shared host).

## 3. Classify (first matching wins, then go to KB section)
| Signal in log | Class | KB |
|---|---|---|
| `libcudart.so.12`, `no kernel image`, `sm_121a`, `ptxas`, `+cpu` torch | ABI / arch | §2.1 |
| `CUDA out of memory`, `NV_ERR_NO_MEMORY`, host freeze, swap growth, killed (OOM killer, `dmesg`) | UMA memory | §2.2 |
| `Image features and image tokens do not match`, shape mismatch in vision path, loss=0 / NaN at step 0 | data / processor | §2.5 |
| `KeyError` / `Missing key(s)` / `Unexpected key(s)` / garbage output after convert | checkpoint | §2.6 |
| NCCL timeout, hang at init, `/dev/shm`, port in use | distributed | §2.4 |
| NaN/inf later in training, quality drop after quantize | numerics | §2.7 + bisect |
Useful: `dmesg -T | tail -50` (read-only), `py-spy dump --pid <PID>` for hangs, `CUDA_LAUNCH_BLOCKING=1` for async CUDA errors, `TORCH_SHOW_CPP_STACKTRACES=1`, `NCCL_DEBUG=INFO`.

## 4. Root cause
- Change ONE variable per run; record each attempt (cmd → result) in `outputs/debug/BUG-NNN/notes.md`.
- Regression in our code → `git bisect run <repro-script>`. Regression from deps → diff `uv pip freeze` between good/bad.
- Write the root cause as one sentence: "X happens because Y, introduced by Z". If you cannot, you are not done.

## 5. Fix
- Write a failing regression test FIRST (`tests/...`, CPU-only if possible; `@pytest.mark.gpu` otherwise). Confirm it fails.
- Minimal diff at the root cause. No drive-by refactors (send those to bao-cleancode instead).
- Fix inside third-party code (NVIDIA libs in a container): never edit in place — a `.patch` under
  `third_party/patches/<lib>/` + `PATCHES.sha256` + `VENDORED.md` (P004 §8.3), every block marked `PATCH(vlm)`, made
  against the exact base *inside the image* (check `git rev-parse HEAD` there, not upstream main). Prove it with a
  test that FAILS on the unpatched image for a behavioural reason (not ImportError of a new helper) and PASSES with the
  patched file bind-mounted. **Send the patch to bao-security and bao-cleancode for review at the same time as to
  bao-backend, and backend builds only after security APPROVE.**
- Workaround (env var / pin / disabled feature) is allowed only if labelled `# WORKAROUND(BUG-NNN): <why> — remove when <condition>` and tracked.
- Run: regression test + module tests + `pytest -m "not gpu and not slow"` + skill `cleancode-check` gates.
- Commit (do NOT push):
  ```
  fix(<module>): <what changed>

  Root cause: <one sentence>
  Repro: <command or test id>
  Fixes BUG-NNN
  ```

## 6. Close the loop
- Update bug file: root cause, fix commit, test id; `Status: FIXED-UNVERIFIED`; write `FIXED <date>` to `outputs/debug/BUG-NNN/STATUS`.
- SendMessage reporter (copy their `from`): "BUG-NNN fixed in <sha>, verify with <test id>". Ask bao-cleancode for review.
- If the bug class is new → add an entry to `.team/knowledge/fixcode.md` §3.
- Log: `HH:MM [bao-fixcode → <reporter>] BUG-NNN fixed <sha>` in `.team/log/<date>.md`.

## Pushback rules
- Report lacks repro/env → ask reporter for it (reference `tester-bug-report` format) rather than guessing.
- "Fix" requires sudo/host change (drop_caches, sysctl, driver) → not ours: write it up and escalate to PM.
