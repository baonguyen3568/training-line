# NeMo-RL v0.7.0 probe image -3 (GRPO, research 2026-10-14). PROBE ONLY until bao-security reviews it.
# -2 plus third_party/patches/nemo-rl/: NVML has no framebuffer memory info on unified-memory GB10
# (nvmlDeviceGetMemoryInfo -> NotSupported), which killed GRPO step 1 at the refit buffer sizing; fall back to
# torch.cuda.mem_get_info. nemo_rl is installed editable from /opt/nemo-rl in every worker venv, so one edit covers all.
# Build from the repo root: docker build -f docker/nemo-rl-probe-3.Dockerfile -t vlm-nemo-rl:0.7.0-3 .
FROM vlm-nemo-rl:0.7.0-2
USER root
COPY third_party/patches/nemo-rl/ /tmp/vlm/patches/
WORKDIR /opt/nemo-rl
RUN set -eu; cd /tmp/vlm/patches && sha256sum -c PATCHES.sha256; cd /opt/nemo-rl; \
    while read -r _ name; do patch -p1 --forward --dry-run < "/tmp/vlm/patches/${name}"; \
      patch -p1 --forward < "/tmp/vlm/patches/${name}"; done < /tmp/vlm/patches/PATCHES.sha256; \
    find /opt/nemo-rl/nemo_rl/utils -name '__pycache__' -prune -exec rm -rf {} +; rm -rf /tmp/vlm
