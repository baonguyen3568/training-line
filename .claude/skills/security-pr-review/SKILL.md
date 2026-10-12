---
name: security-pr-review
description: Use when reviewing any change (diff, branch, PR, or uncommitted files) that touches Dockerfile/compose, pyproject/uv.lock/requirements, network/ports/serving, model or dataset loading, env/secrets handling, or scripts that delete/move files — and before the first commit or any commit to run the secret scan. Produces a security verdict with blocking vs non-blocking findings.
---

# Security review of a change

Knowledge: `.team/knowledge/security.md` (§9 checklist). Be evidence-based: every finding cites file:line and a concrete failure scenario.

## 1. Scope the diff
```bash
cd /home/ntcai/Trainning-model
git status --short; git diff --stat HEAD 2>/dev/null || git ls-files -o --exclude-standard
git diff HEAD -- 'docker/**' '**/Dockerfile*' '**/*compose*.y*ml' pyproject.toml uv.lock 'requirements*' '.env*' 'scripts/**' 'src/**'
```

## 2. Secrets (always, blocking)
```bash
# preferred: pre-commit hook; ad-hoc without install:
docker run --rm -v "$PWD":/repo:ro zricethezav/gitleaks:<pinned-tag> dir /repo --redact -v
uvx detect-secrets scan --all-files --exclude-files '(^|/)(\.venv|data|outputs|checkpoints)/'
git ls-files | grep -Ei '(^|/)\.env$|\.pem$|id_rsa|token|credentials' || true
grep -rnE '(hf_[A-Za-z0-9]{30,}|nvapi-[A-Za-z0-9_-]{20,}|WANDB_API_KEY\s*=\s*\S{10,})' --exclude-dir={.git,.venv,data,outputs} . || true
```
Also: no `ENV/ARG *TOKEN*|*KEY*` in Dockerfile (use `RUN --mount=type=secret`); no logging of `os.environ`; W&B config free of tokens.
Leak found ⇒ BLOCK + tell owner to **revoke** the token, notify PM. Never paste the secret in messages/logs.

## 3. Docker / compose (blocking unless noted)
- [ ] base image `@sha256:` and listed in `docker/IMAGES.lock`; arm64
- [ ] no `--privileged`, `network_mode: host`, `pid: host`, `ipc: host` (non-blocking if justified), `cap_add` without reason
- [ ] no mount of `/`, `$HOME`, `~/.ssh`, `/var/run/docker.sock`, `~/.cache/huggingface/token` (mount `hub/` only)
- [ ] `ports` bound to `127.0.0.1:` only
- [ ] `user:` non-root (or documented reason), `security_opt: [no-new-privileges:true]`, `cap_drop: [ALL]`
- [ ] `mem_limit` set (GB10 unified memory: container RAM = host RAM shared with other services)
- [ ] container/image name prefix `vlm-`, label `project=vlm-training`
- [ ] no `LD_PRELOAD`, no `curl | sh`, apt pins where practical, `--no-install-recommends`
- Lint: `docker run --rm -i hadolint/hadolint:<pinned> < docker/Dockerfile`

## 4. Python deps
- [ ] `uv.lock` updated & committed with pyproject; install path uses `uv sync --locked`
- [ ] new package: legit name (typosquat check on pypi.org: maintainer, age, downloads), wheel available for aarch64
- [ ] no new `--extra-index-url`; extra indexes declared as `[[tool.uv.index]]` with `explicit = true` + per-package `[tool.uv.sources]`
- [ ] no `git+...@main` (pin commit)
- [ ] `exclude-newer` cooldown respected
- [ ] `uvx pip-audit` (in the venv) → no CRITICAL/HIGH with available fix (non-blocking otherwise; record)

## 5. Model / data loading code
- [ ] `trust_remote_code` default False; True only for allowlisted repo+sha (see `security-model-vetting`)
- [ ] `revision=` is a 40-char SHA from config, not hardcoded `main`
- [ ] `use_safetensors=True`; no `torch.load`/`pickle.load`/`np.load(allow_pickle=True)` on external files; internal `torch.load` uses `weights_only=True` and torch ≥ 2.6
- [ ] image decode: `Image.MAX_IMAGE_PIXELS` set, decode errors handled, no fetching arbitrary URLs from dataset rows without allowlist
- [ ] ruff `S` rules pass (S301 pickle, S307 eval, S603/S605 subprocess, S108 tmp, S113 requests timeout)

## 6. Serving / network
- [ ] bind 127.0.0.1, API key on (vLLM `--api-key`), remote media URL fetching disabled unless needed
- [ ] vLLM/TRT-LLM version ≥ latest security fix (check GitHub advisories)

## 7. File-system safety (scripts)
- [ ] deletes resolve `realpath` inside repo; dry-run default; no `rm -rf "$VAR"/` with possibly-empty var (`set -u`)
- [ ] no `sudo`, no `docker system prune`, docker ops filtered by `name=vlm-` / label

## Verdict
Reply to author (SendMessage) + log line:
```
SEC-REVIEW <branch/PR/files>: APPROVE | REQUEST-CHANGES
BLOCKING: - file:line — issue — scenario — fix
NON-BLOCKING: - ...
```
Disagreement not resolved in 1 round → `.team/CONFLICTS.md` + notify PM (`trainning-model-6a`).

## Ordering rule (learned 2026-10-12)
- Third-party code patches (`third_party/patches/**`) and any deviation from an approved lock/Dockerfile must reach
  security **before** the image is built. If a build already happened, review retroactively, compare the built
  artifact (pip freeze, patch sha256, image digest) against what was reviewed, and record the deviation in the log.
- Always verify by artifact, not by report: recompute sha256 of the exact file built, diff pinned hash lists, read
  the committed code (not the description).
