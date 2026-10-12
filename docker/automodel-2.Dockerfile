# vlm-automodel:26.08-N = NGC 26.08 (chmod o+rX) + pinned extra wheels + fixcode's AutoModel patches (PATCHES.sha256).
# N = number of reviewed patch sets; pass --build-arg VLM_TAG=26.08-N matching the -t tag.
# Build from the repo root with the matching .dockerignore (only docker/automodel/ and third_party/patches/automodel/
# are sent):  docker buildx build --builder vlm-builder --platform linux/arm64 \
#               -f docker/automodel-2.Dockerfile -t vlm-automodel:26.08-2 --load .
# Base pinned by digest (arm64 manifest of nemo-automodel:26.08); same chmod as 26.08-1.
FROM nvcr.io/nvidia/nemo-automodel@sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a

ARG VLM_TAG=26.08-3
LABEL project="vlm-training" \
      org.opencontainers.image.title="vlm-automodel" \
      org.opencontainers.image.version="${VLM_TAG}" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/nemo-automodel:26.08" \
      org.opencontainers.image.base.digest="sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a"

# No phoning home from libraries (bao-security, pre-approved). Jobs needing W&B online override WANDB_MODE explicitly.
ENV NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    WANDB_MODE=offline

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) Extra wheels needed by Nemotron-Parse remote processor code (albumentations, cv2). Hash-locked, no deps
#    resolution at build time (versions of numpy/scipy/pydantic/pyyaml are constrained to what the image has).
COPY docker/automodel/extras.lock.txt /tmp/vlm/extras.lock.txt
#    The base image already has unrelated pip-check conflicts (torchx/urllib3, leptonai): fail only on NEW ones.
RUN (/opt/venv/bin/python -m pip check > /tmp/vlm/check.before 2>&1 || true) \
 && /opt/venv/bin/python -m pip install --isolated --no-cache-dir --index-url https://pypi.org/simple \
      --only-binary=:all: --require-hashes --no-deps -r /tmp/vlm/extras.lock.txt \
 && (/opt/venv/bin/python -m pip check > /tmp/vlm/check.after 2>&1 || true) \
 && diff /tmp/vlm/check.before /tmp/vlm/check.after \
 && /opt/venv/bin/python -c "import albumentations, cv2; print(albumentations.__version__, cv2.__version__, cv2.__file__)" \
 && [[ "$(/opt/venv/bin/python -m pip list 2>/dev/null | grep -ciE '^opencv-')" -eq 1 ]]

# 2) AutoModel source patches (bao-fixcode). Fail the build unless hashes match and every patch applies cleanly.
COPY third_party/patches/automodel/ /tmp/vlm/patches/
#    Only patches listed in PATCHES.sha256 are applied, in that order; any unlisted *.patch fails the build.
WORKDIR /tmp/vlm/patches
RUN sha256sum -c PATCHES.sha256 \
 && diff <(awk '{print $2}' PATCHES.sha256 | sort) <(find . -maxdepth 1 -name '*.patch' -printf '%f\n' | sort)
WORKDIR /opt/Automodel
RUN applied=0; \
    while read -r _ name || [[ -n "${name}" ]]; do \
      patch -p1 --forward --dry-run < "/tmp/vlm/patches/${name}" && patch -p1 --forward < "/tmp/vlm/patches/${name}" \
      && applied=$((applied + 1)); \
    done < /tmp/vlm/patches/PATCHES.sha256; \
    expected=$(grep -c . /tmp/vlm/patches/PATCHES.sha256); \
    [[ "${applied}" -eq "${expected}" ]] || { echo "applied ${applied}/${expected} patches" >&2; exit 1; }; \
    rm -rf /tmp/vlm

RUN chmod -R o+rX /opt/Automodel
USER 1000:1000
