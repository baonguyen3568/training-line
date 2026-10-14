#!/usr/bin/env bash
# Integration gates inside the NGC-derived image (ADR-004; owner bao-backend). Called by `scripts/check.sh --container`.
#   scripts/container/check.sh [--image IMG] [--gpu]   (default IMG = common.images.DEFAULT_TRAIN_IMAGE)
# Without --gpu: import smoke + `pytest -m "integration and not gpu"` (no GPU, no lock, network none, repo read-only).
# With --gpu: also `pytest -m gpu` under the GPU lock (token passed as VLM_GPU_LOCK_TOKEN; tests verify it).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
IMAGE=$(uv run --quiet python -c "from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE; print(DEFAULT_TRAIN_IMAGE)")
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --gpu) GPU=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found (see docker/IMAGES.lock)" >&2; exit 2; }

GATE_N=0
GATE_MEM=${VLM_GATE_MEM:-24g}
[[ $GATE_MEM =~ ^[0-9]+[gGmM]$ ]] || { echo "VLM_GATE_MEM must look like 24g or 16384m, got '$GATE_MEM'" >&2; exit 2; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run() {  # run a command in a throw-away, locked-down container with the repo mounted read-only
  # Named vlm-* like every project container (CLAUDE.md: prefix vlm-), so it is never mistaken for another service's.
  # Memory = swap: a gate never swaps on the shared host (it held 4.3 GB of swap and queued a GPU job, 2026-10-14).
  # 24 GB: the full integration shard peaked at ~16 GB (11.3 GiB RAM + 4.3 GB swap) on vlm-automodel:26.08-8.
  # Not a GPU memory guard: on GB10 (UMA) CUDA allocations may sit outside the cgroup limit; GPU jobs use memguard.
  docker run --rm --name "vlm-gate-$$-$((++GATE_N))" --label project=vlm-training --network none --cap-drop ALL \
    --security-opt no-new-privileges --memory "$GATE_MEM" --memory-swap "$GATE_MEM" \
    --pids-limit 1024 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/workspace/src -e VLM_PROJECT_ROOT=/workspace \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/workspace/.cache/hf -e HF_DATASETS_CACHE=/tmp/hf_datasets \
    -e HF_DATASETS_OFFLINE=1 -e NO_ALBUMENTATIONS_UPDATE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$ROOT:/workspace:ro" -w /workspace "$@"
}
PYTEST=(python -m pytest -p no:cacheprovider -q --import-mode=importlib)

step "import smoke ($IMAGE)"
run "$IMAGE" python -c "import vlm_pipeline, vlm_pipeline.common.gpu_lock, vlm_pipeline.common.runinfo, vlm_pipeline.i18n; print('ok')"

step "integration tests (no GPU)"
rc=0
run "$IMAGE" "${PYTEST[@]}" -m "integration and not gpu" tests/integration || rc=$?
[[ $rc -eq 0 || $rc -eq 5 ]] || exit $rc  # 5 = no tests collected

if [[ $GPU -eq 1 ]]; then
  step "GPU tests (under GPU lock)"
  LOCK_JSON=$(uv run --quiet python -m vlm_pipeline.common.gpu_lock acquire --who "${USER:-dev}" \
    --job "check.sh --container --gpu" --run-id "check-$(date +%s)" --host-pid $$) || { echo "$LOCK_JSON"; exit 4; }
  LOCK_TOKEN=$(python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])' <<<"$LOCK_JSON")
  trap 'uv run --quiet python -m vlm_pipeline.common.gpu_lock release --token "$LOCK_TOKEN"' EXIT
  rc=0
  run --gpus all -e "VLM_GPU_LOCK_TOKEN=$LOCK_TOKEN" "$IMAGE" "${PYTEST[@]}" -m gpu tests/integration || rc=$?
  [[ $rc -eq 0 || $rc -eq 5 ]] || exit $rc
fi
printf '\n\033[32mContainer gates passed (%s).\033[0m\n' "$IMAGE"
