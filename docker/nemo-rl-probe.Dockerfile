# NeMo-RL v0.7.0 probe image (GRPO VLM probe, research 2026-10-14; bao-security approved this exact change).
# The upstream venv's python is a symlink into /root/.local/share/uv/python (755), but /root itself is 700, so a
# non-root --user cannot execute python. Only /root's traverse bit changes (711: no listing); copy-up is ~0.
# PROBE ONLY, not approved for the app (IMAGES.lock). Never push or share this image (NVIDIA DL Container License).
FROM nvcr.io/nvidia/nemo-rl@sha256:b07327990c157877d84c91e53294dd95eed9442de892e7b8f0e8eda5c9425c25
RUN chmod 711 /root
