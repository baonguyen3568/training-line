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

## 2. Acquire lock (atomic; fails if someone else holds it)
```bash
python3 - <<'PY'
import json, os, time, sys
p = ".team/GPU_LOCK"
try:
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
except FileExistsError:
    sys.exit("GPU_LOCK held: " + open(p).read())
os.write(fd, json.dumps({"who": "bao-backend", "job": "JOB_DESC", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "host_pid": os.getppid(), "container": "vlm-JOBNAME"}, indent=1).encode()); os.close(fd)
PY
```
`host_pid` must be the HOST PID of the process that holds the lock for the whole job (the orchestrator / your shell), never a PID inside the container — the dashboard uses it for stale detection. Once `common/gpu_lock.py` exists (P008), use it instead of this snippet (tmp file + os.link, token).
Tell bao-tester (they sample /proc/meminfo during spikes) and log a line in `.team/log/<date>.md`.

## 3. Run the container (conventions)
```bash
RUN_ID=$(date +%Y%m%d-%H%M)-<recipe>-<model>-$(head -c2 /dev/urandom | xxd -p)
RUN_DIR=outputs/runs/$RUN_ID; mkdir -p $RUN_DIR/{logs,checkpoints,export} .cache/hf
docker run --rm --name vlm-$RUN_ID --label project=vlm-training \
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
```bash
rm -f .team/GPU_LOCK      # only if it is ours: check "who" first
```
Record peak memory, s/step, tok/s, digest into run.json / the proposal; log to `.team/log/<date>.md`.
