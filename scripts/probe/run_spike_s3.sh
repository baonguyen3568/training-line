#!/usr/bin/env bash
# S-3 GPU smoke launcher (temporary, until the job runner of P008 exists).
# Holds .team/GPU_LOCK for its whole lifetime (host_pid = this script), runs AutoModel in vlm-automodel,
# records exit code / OOMKilled, releases the lock on exit. Never touches other people's locks.
set -euo pipefail
ROOT=/home/ntcai/Trainning-model; cd "$ROOT"
IMAGE=vlm-automodel:26.08-1
CFG_SRC=configs/train/smoke/qwen3_5_0p8b_lora_medpix_spike.yaml
EST_PEAK_GB=${EST_PEAK_GB:-12}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)-lora-qwen3.5-0.8b-$(head -c2 /dev/urandom | od -An -tx1 | tr -d ' \n')}
RUN_DIR=outputs/runs/$RUN_ID; NAME=vlm-$RUN_ID

avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if (( avail_gb < EST_PEAK_GB + 8 )); then echo "PREFLIGHT FAIL: MemAvailable ${avail_gb}GB < ${EST_PEAK_GB}+8"; exit 2; fi

python3 - "$RUN_ID" "$$" <<'PY'
import json, os, sys, time, secrets
p = ".team/GPU_LOCK"; tmp = f"{p}.tmp.{os.getpid()}"
with open(tmp, "w") as f:
    json.dump({"who": "bao-backend", "job": "S-3 LoRA 20 steps Qwen3.5-0.8B (AutoModel 26.08)", "run_id": sys.argv[1],
               "host_pid": int(sys.argv[2]), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "token": secrets.token_hex(16)}, f, indent=1)
try:
    os.link(tmp, p)
except FileExistsError:
    os.unlink(tmp); sys.exit("GPU_LOCK busy: " + open(p).read())
os.unlink(tmp)
PY
release() { python3 -c "import json,sys;d=json.load(open('.team/GPU_LOCK'));sys.exit(0 if d.get('run_id')=='$RUN_ID' else 1)" && rm -f .team/GPU_LOCK; }
trap release EXIT

mkdir -p "$RUN_DIR"/{logs,checkpoints}
sed "s#RUN_DIR#/workspace/$RUN_DIR#g" "$CFG_SRC" > "$RUN_DIR/config.yaml"
echo "RUN_ID=$RUN_ID NAME=$NAME"
start=$(date +%s)
set +e
docker run --name "$NAME" --label project=vlm-training \
  --gpus all --shm-size=8g --network none --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt no-new-privileges --cap-drop ALL --memory=48g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e HF_HOME=/workspace/.cache/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_CACHE=/workspace/$RUN_DIR/hf_datasets_cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TRITON_CACHE_DIR=/tmp/triton \
  -v "$ROOT:/workspace" -w /opt/Automodel \
  "$IMAGE" automodel "/workspace/$RUN_DIR/config.yaml" --nproc-per-node 1 > "$RUN_DIR/logs/stdout.log" 2>&1
rc=$?
set -e
end=$(date +%s)
state=$(docker inspect -f '{"exit_code":{{.State.ExitCode}},"oom_killed":{{.State.OOMKilled}}}' "$NAME")
docker rm "$NAME" >/dev/null
echo "{\"run_id\":\"$RUN_ID\",\"image\":\"$IMAGE\",\"docker_rc\":$rc,\"state\":$state,\"wall_s\":$((end-start))}" | tee "$RUN_DIR/result.json"
