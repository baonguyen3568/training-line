# NeMo-RL v0.7.0 probe image -2 (GRPO probe, research 2026-10-14; bao-security chose this variant). PROBE ONLY.
# NeMo-RL's vLLM worker patches three vLLM sources at start (nemo_rl/models/generation/vllm/patches.py) under an
# open(<file>.patch_lock, "w") + flock. Here the patch is applied ONCE at build time by NeMo-RL's own function (the
# four worker venvs share one vLLM copy), so at run time it finds "already patched" and writes no source. Only the
# lock files exist and are group-writable (run as --user uid:gid --group-add 0); no directory or .py is made writable.
FROM vlm-nemo-rl:0.7.0-1
RUN set -eu; \
    W=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker; \
    cd /opt/nemo-rl && "$W/bin/python" -c "from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches; _apply_vllm_patches('$W/bin/python')"; \
    for v in /opt/ray_venvs/*; do d="$v/lib/python3.13/site-packages/vllm"; [ -d "$d" ] || continue; \
      for f in v1/executor/ray_executor.py model_executor/models/llama_eagle3.py tool_parsers/hermes_tool_parser.py; do \
        touch "$d/$f.patch_lock"; chmod 664 "$d/$f.patch_lock"; done; done
