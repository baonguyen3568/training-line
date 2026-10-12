---
name: cleanup-disk-audit
description: Use to report disk/RAM usage of the VLM project on the shared DGX Spark (before a big download, before/after training, when someone asks "how much space/memory is left", or on a periodic health check). Read-only — never deletes anything.
---

# Disk & memory audit (read-only)

Project root: `/home/ntcai/Trainning-model`. Host is SHARED — report only, never act on objects outside the project.

## 1. Host level
```bash
df -h /                                   # alert PM if Avail < 300G
free -g; grep -E 'MemAvailable|SwapFree' /proc/meminfo   # GB10 unified memory: nvidia-smi shows N/A
cat /home/ntcai/Trainning-model/.team/GPU_LOCK 2>/dev/null   # who is running heavy job
```
Thresholds: MemAvailable < 20G → warn `trainning-model-6a` (PM); < 10G → recommend pausing *our* job. Never touch other people's processes.

## 2. Project footprint
```bash
cd /home/ntcai/Trainning-model
du -xsh .cache data outputs checkpoints 2>/dev/null
du -xh --max-depth=2 outputs data 2>/dev/null | sort -h | tail -20
find . -xdev -type f -size +1G -not -path './.git/*' -printf '%s\t%TY-%Tm-%Td\t%p\n' 2>/dev/null | sort -rn | numfmt --field=1 --to=iec | head -30
```

## 3. HF cache (project cache only)
```bash
uvx --from huggingface_hub hf cache ls --cache-dir .cache/hf/hub --sort size:desc
uvx --from huggingface_hub hf cache ls --cache-dir .cache/hf/hub --revisions --filter "accessed>30d"
du -sh .cache/hf/datasets .cache/hf/xet 2>/dev/null   # not counted by hf cache ls
```
Global `~/.cache/huggingface` (~190G) belongs to other users/projects — may be *listed* for reference, never cleaned.

## 4. Checkpoints
```bash
for d in outputs/runs/*/checkpoints; do echo "== $d"; ls -l "$d" | grep -E 'LATEST|LOWEST_VAL'; du -sh "$d"/epoch_* 2>/dev/null; done
du -sh outputs/runs/*/checkpoints/*/optim 2>/dev/null   # optimizer state ≈ 2x weights, first thing to strip
```

## 5. Docker (project objects only)
```bash
docker ps -a --filter name=^vlm- --format '{{.Names}}\t{{.Status}}\t{{.Size}}'
docker images --filter label=project=vlm-training --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep -E '^vlm-'
docker volume ls --filter label=project=vlm-training
docker buildx du --builder vlm-builder 2>/dev/null | tail -3
```
`docker system df` is OK to read, but its "reclaimable" numbers are for the whole host — don't quote them as ours.

## 6. Output
Write a short table (item / size / age / proposed action) and, if notable, append one line to `.team/log/YYYY-MM-DD.md`:
`HH:MM [bao-cleanup → bao-pm] disk report: project X GB, free Y TB, MemAvailable Z GB`.
