---
name: cleanup-safe-delete
description: Use whenever deleting anything for the VLM project — old checkpoints, HF cache entries, datasets, outputs, vlm-* docker containers/images/build cache — or applying the checkpoint retention policy. Enforces scope, dry-run first, and PM/security sign-off for big deletes.
---

# Safe delete & retention

## Hard rules (from CLAUDE.md — non-negotiable)
- Scope = paths that `realpath` under `/home/ntcai/Trainning-model`, or docker objects with name `vlm-*` AND/OR label `project=vlm-training`.
- FORBIDDEN: `docker system prune`, `docker image prune -a` without label filter, `docker builder prune` on the default builder, removing anything in `~/.cache/huggingface` (global), `sudo`, `rm -rf` outside project.
- Dry-run first, always. Real delete needs explicit `--apply`.
- > 20GB, or anything tagged best/final/release, or ambiguous → ask PM (`trainning-model-6a`) + `bao-security` first, wait for OK.

## Pre-delete checklist
1. `realpath <target>` starts with `/home/ntcai/Trainning-model/` (guards against symlinks escaping).
2. No `KEEP` marker file in target or its run dir.
3. Not the target of `LATEST` / `LOWEST_VAL` / `best` symlink: `readlink -f outputs/runs/<run>/checkpoints/{LATEST,LOWEST_VAL}`.
4. Not in use: `cat .team/GPU_LOCK`; `fuser -vm <target>` / `lsof +D <target>` empty; no `*.tmp` / partial async save.
5. Size known (`du -sh`), dry-run listing reviewed.
6. Log after: `HH:MM [bao-cleanup → bao-pm] deleted <what> (<size>)` in `.team/log/YYYY-MM-DD.md`.

## Checkpoint retention (default, see Proposal 005)
During training (NeMo AutoModel): `checkpoint.max_recent_checkpoints: 2`, `checkpoint.save_consolidated: final`.
After a run finishes:
- keep: LATEST, LOWEST_VAL/best, consolidated final export, all LoRA adapters (small).
- strip `optim/` (and `dataloader/`, `rng/`) from every checkpoint that is not LATEST.
- delete other `epoch_*_step_*` dirs older than 7 days unless `KEEP` present.
NeMo-RL: `checkpointing.keep_top_k`. Megatron/Lightning: `save_top_k` + `save_last`.

## Commands
```bash
# Shell hygiene for any ad-hoc script
set -euo pipefail; ROOT=/home/ntcai/Trainning-model
t=$(realpath -e "${TARGET:?}"); [[ "$t" == "$ROOT"/* ]] || { echo "OUT OF SCOPE: $t"; exit 1; }

# HF cache (project cache only) — never rm -rf models--* (shared xet blobs)
uvx --from huggingface_hub hf cache prune --cache-dir $ROOT/.cache/hf/hub --dry-run
uvx --from huggingface_hub hf cache rm model/<org>/<name> --cache-dir $ROOT/.cache/hf/hub --dry-run   # then -y
rm -rf "$ROOT/.cache/hf/xet"   # upload-only shard cache, safe

# Docker — project objects only
docker ps -a --filter name=^vlm- --filter status=exited          # review
docker container prune --filter label=project=vlm-training --filter until=72h
docker image prune --filter label=project=vlm-training --filter until=168h   # dangling only
docker images --filter label=project=vlm-training                 # tagged: docker rmi one by one
docker ps -a --filter ancestor=<image>                            # make sure no non-project container uses it
docker buildx prune --builder vlm-builder --filter until=168h
```
Project tools (built, reviewed):
```bash
uv run python scripts/cleanup/disk_report.py                      # footprint + levels -> outputs/system/disk_report.json
uv run python scripts/cleanup/preflight.py --est-peak-gb N --ckpt-gb M   # before taking GPU_LOCK (exit 0/2/3/4)
uv run python scripts/cleanup/retention.py [--run ID] [--json]    # checkpoint retention DRY-RUN
uv run python scripts/cleanup/retention.py --apply --by bao-cleanup [--approved-by bao-pm]   # >20 GiB/24h needs approval
```
retention only touches `outputs/runs/*/checkpoints/epoch_*`; anything else (misplaced caches, smoke dirs) = manual delete
with the checklist above + PM approval, then append an event to `outputs/system/cleanup.jsonl` and `.team/log`.
Code changes: commit only AFTER review (CLAUDE.md), `git commit -- <your paths>` only.

## Memory (GB10 unified)
Do NOT run `drop_caches` (needs sudo, host-wide) — ask PM. Never kill processes not started by this project.
