# Derived image: NeMo AutoModel 26.08 readable by non-root (uid 1000 / plugin uid 61001).
# Upstream ships /opt/Automodel/nemo_automodel as root-only (drwxrwx---), which breaks the
# non-root runtime baseline (P003). Only change: grant others read + traverse (no write).
# Base pinned to the linux/arm64 manifest digest (index digest: sha256:b0bd0e50…3aee).
FROM nvcr.io/nvidia/nemo-automodel@sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a

LABEL project="vlm-training" \
      org.opencontainers.image.title="vlm-automodel" \
      org.opencontainers.image.version="26.08-1" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/nemo-automodel:26.08" \
      org.opencontainers.image.base.digest="sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a"

USER root
RUN chmod -R o+rX /opt/Automodel
USER 1000:1000
