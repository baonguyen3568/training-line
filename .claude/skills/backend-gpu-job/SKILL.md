---
name: backend-gpu-job
description: Use before launching ANY heavy job (train / quantize / eval / benchmark / serve) on the shared DGX Spark GB10 for this project — acquiring/releasing .team/GPU_LOCK, checking the unified-memory budget, and running the NGC container with the project's docker conventions (vlm-* name, non-root, project-local caches, pinned digest, run dir layout).
---
# Heavy GPU job on the shared GB10 (bao-backend)

Rules come from CLAUDE.md ("GPU lock", "Quy tắc an toàn"). Never kill or delete other people's jobs/locks/containers.

## 1. Pre-flight (read-only)
```bash
cd /home/ntcai/Trainning-model
cat .team/GPU_LOCK 2>/dev/null || echo "lock free"
grep -E 'MemTotal|MemAvailable|SwapFree' /proc/meminfo      # nvidia-smi shows no memory on GB10
docker ps --format '{{.Names}}\t{{.Image}}' | grep '^vlm-' || true
df -h / | tail -1
```
- Lock present: if its PID is alive (`kill -0 <pid>`), WAIT / ask owner. If PID is dead → message the owner + bao-pm, do NOT delete it yourself.
- Budget: require `MemAvailable >= expected_peak + 8 GB`. Reference peaks (GB10, AutoModel): 8B LoRA ≈ 21 GiB, 14B LoRA ≈ 43 GiB, 8B full SFT ≈ 78 GiB.
  Shared host usually leaves only ~40 GB → default to LoRA ≤ 8B / full SFT ≤ 2-4B.
- Downloads > 20 GB → tell bao-pm first. New model/image → `security-model-vetting` skill.

## 2. Acquire lock — ALWAYS via `common/gpu_lock.py` (one format for everyone; preflight/dashboard parse it)
```bash
uv run python scripts/cleanup/preflight.py --est-peak-gb N --ckpt-gb M || exit $?   # 0 ok, 2 mem, 3 disk, 4 lock busy
LOCK_JSON=$(uv run python -m vlm_pipeline.common.gpu_lock acquire \   # stdout holds the token: never tee it into run logs
  --who bao-backend --job "what you run" --run-id "$RUN_ID" --host-pid $$) || { echo "$LOCK_JSON"; exit 4; }
LOCK_TOKEN=$(python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])' <<<"$LOCK_JSON")
trap 'uv run python -m vlm_pipeline.common.gpu_lock release --token "$LOCK_TOKEN"' EXIT
uv run python -m vlm_pipeline.common.gpu_lock status     # exit 4 + JSON (no token) while held
```
- `--host-pid` = PID of the process that stays alive for the whole job (the launcher shell `$$`), never a container PID.
- Never write the lock by hand (`echo ... > GPU_LOCK`): a hand-written lock is "malformed" for everyone else.
- Pass the token to containers as `-e VLM_GPU_LOCK_TOKEN=$LOCK_TOKEN` (gpu pytest + worker check it); mount the repo so `.team/GPU_LOCK` is visible.
- Tell bao-tester (they sample /proc/meminfo + cgroup) ≥60 s before starting; log a line in `.team/log/<date>.md`.

## 3. Run the container (conventions)
```bash
RUN_ID=$(date +%Y%m%d-%H%M)-<recipe>-<model>-$(head -c2 /dev/urandom | xxd -p)
RUN_DIR=outputs/runs/$RUN_ID; mkdir -p $RUN_DIR/{logs,checkpoints,export} .cache/hf
docker run --name vlm-$RUN_ID --label project=vlm-training \
  --gpus all --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt no-new-privileges --cap-drop ALL --memory=48g \
  --user $(id -u):$(id -g) -e HOME=/tmp \
  --env-file .env -e HF_HOME=/workspace/.cache/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -v $PWD:/workspace -w /workspace \
  nvcr.io/nvidia/<image>@sha256:<digest> \
  <command> 2>&1 | tee $RUN_DIR/logs/stdout.log
```
- Security baseline (bao-security, CVE-2026-24260 on host toolkit): nvcr.io images pinned @sha256 only, no `--privileged`, no `--network host`,
  no docker.sock / $HOME mounts, ports `-p 127.0.0.1:787x:787x`, tokens only via `--env-file .env`. Use `--shm-size` not `--ipc=host`.
- `--memory` caps CPU RSS only; it does NOT cap CUDA allocations on UMA — limit memory via batch/seq len, `gpu_memory_utilization` (vLLM ≤ 0.3 on this shared host), and the in-code memguard.
- Pin by digest: `docker inspect --format '{{index .RepoDigests 0}}' <image:tag>`; record it in run.json.
- If the image refuses non-root, run as root and `chown -R $(id -u):$(id -g) $RUN_DIR` afterwards (inside project only).
- AutoModel prints loss/mem/tps on stderr → always `2>&1`.

## 4. During the run
`watch -n5 "grep MemAvailable /proc/meminfo"`. If MemAvailable < 6 GB or swap climbs fast → stop OUR job (`docker stop vlm-$RUN_ID`), never others.

## 5. Release
The `trap` above releases with the token (only removes the lock if it is still ours).
Record peak memory, s/step, tok/s, digest into run.json / the proposal; log to `.team/log/<date>.md`.
