#!/usr/bin/env bash
# Integration gates inside the NGC-derived image (ADR-004; owner bao-backend). Called by `scripts/check.sh --container`.
#   scripts/container/check.sh [--image IMG] [--gpu] [--rev REV | --dirty]   (default IMG = DEFAULT_TRAIN_IMAGE)
# Without --gpu: import smoke + `pytest -m "integration and not gpu"` (no GPU, no lock, network none, repo read-only).
# With --gpu: also `pytest -m gpu` under the GPU lock (token passed as VLM_GPU_LOCK_TOKEN; tests verify it).
# Code under test: a pristine worktree of REV (default HEAD), like check.sh --clean and install.sh: the shared tree
# changes under a 25-minute run (2026-10-14: a commit landed mid-run and one test saw half-old files). Only the
# untracked inputs come from the shared tree, read-only: .cache/hf, data/, outputs/. --dirty mounts the shared
# tree as before (to try uncommitted work; never the evidence for an image).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
IMAGE=""  # default: DEFAULT_TRAIN_IMAGE of the code under test (read after the worktree exists)
GPU=0
REV=HEAD
DIRTY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --gpu) GPU=1; shift ;;
    --rev) REV=$2; shift 2 ;;
    --dirty) DIRTY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
MOUNTS=(-v "$ROOT:/workspace:ro")
cleanup() { :; }
if [[ $DIRTY -eq 0 ]]; then
  COMMIT=$(git rev-parse --verify --quiet "$REV^{commit}") || { echo "--rev $REV is not a commit" >&2; exit 2; }
  [[ ! -L "$ROOT/.cache" ]] || { echo ".cache must not be a symlink" >&2; exit 2; }
  mkdir -p "$ROOT/.cache"
  # Worktrees left by gate runs that died (the name carries the owner PID; a live run's tree is never touched).
  for stale in "$ROOT"/.cache/gate-clean.*; do
    [[ -d "$stale" && ! -L "$stale" ]] || continue
    owner="${stale##*/gate-clean.}"; owner="${owner%%.*}"
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then continue; fi
    git worktree remove --force "$stale" >/dev/null 2>&1 || rm -rf --one-file-system "$stale"
  done
  git worktree prune >/dev/null 2>&1 || true
  WORKTREE="$(mktemp -d "$ROOT/.cache/gate-clean.$$.XXXXXX")"
  cleanup() {
    git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    git worktree prune >/dev/null 2>&1 || true
    rm -rf --one-file-system "$WORKTREE" 2>/dev/null || true  # only the mktemp dir created above
  }
  trap cleanup EXIT
  trap 'cleanup; exit 130' INT
  trap 'cleanup; exit 143' TERM HUP
  git worktree add --detach --quiet "$WORKTREE" "$COMMIT"
  mkdir -p "$WORKTREE/.cache/hf" "$WORKTREE/data" "$WORKTREE/outputs"  # mount points inside the read-only tree
  MOUNTS=(-v "$WORKTREE:/workspace:ro")
  for d in .cache/hf data outputs; do
    [[ -d "$ROOT/$d" ]] && MOUNTS+=(-v "$ROOT/$d:/workspace/$d:ro")
  done
  CODE="$(git rev-parse --short "$COMMIT") (pristine worktree; untracked inputs read-only)"
  SRC="$WORKTREE/src"
else
  CODE="the SHARED working tree (--dirty): not evidence for an image"
  SRC="$ROOT/src"
fi
# The image the code under test names as its default (images.py imports nothing heavy), unless --image was given.
[[ -n "$IMAGE" ]] || IMAGE=$(PYTHONPATH="$SRC" python3 -c \
  "from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE; print(DEFAULT_TRAIN_IMAGE)")
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null) \
  || { echo "image $IMAGE not found (see docker/IMAGES.lock)" >&2; exit 2; }
EVIDENCE="code under test: $CODE; image $IMAGE = $IMAGE_ID"
echo "$EVIDENCE"

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
    "${MOUNTS[@]}" -w /workspace "$@"
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
  trap 'uv run --quiet python -m vlm_pipeline.common.gpu_lock release --token "$LOCK_TOKEN"; cleanup' EXIT
  rc=0
  run --gpus all -e "VLM_GPU_LOCK_TOKEN=$LOCK_TOKEN" "$IMAGE" "${PYTEST[@]}" -m gpu tests/integration || rc=$?
  [[ $rc -eq 0 || $rc -eq 5 ]] || exit $rc
fi
printf '\n\033[32mContainer gates passed.\033[0m %s\n' "$EVIDENCE"
