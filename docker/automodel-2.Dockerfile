# vlm-automodel:26.08-2 = 26.08-1 (chmod o+rX) + pinned extra wheels + fixcode's AutoModel patches.
# Build from the repo root with the matching .dockerignore (only docker/automodel/ and third_party/patches/automodel/
# are sent):  docker buildx build --builder vlm-builder --platform linux/arm64 \
#               -f docker/automodel-2.Dockerfile -t vlm-automodel:26.08-2 --load .
# Base pinned by digest (arm64 manifest of nemo-automodel:26.08); same chmod as 26.08-1.
FROM nvcr.io/nvidia/nemo-automodel@sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a

LABEL project="vlm-training" \
      org.opencontainers.image.title="vlm-automodel" \
      org.opencontainers.image.version="26.08-2" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/nemo-automodel:26.08" \
      org.opencontainers.image.base.digest="sha256:70b4e9df8dd23ef451372b25952cffe24ad29c17f5b9de5fd1b244caf4e7b55a"

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) Extra wheels needed by Nemotron-Parse remote processor code (albumentations, cv2). Hash-locked, no deps
#    resolution at build time (versions of numpy/scipy/pydantic/pyyaml are constrained to what the image has).
COPY docker/automodel/extras.lock.txt /tmp/vlm/extras.lock.txt
RUN /opt/venv/bin/python -m pip install --isolated --no-cache-dir --index-url https://pypi.org/simple \
      --only-binary=:all: --require-hashes --no-deps -r /tmp/vlm/extras.lock.txt \
 && /opt/venv/bin/python -m pip check \
 && /opt/venv/bin/python -c "import albumentations, cv2; print(albumentations.__version__, cv2.__version__)"

# 2) AutoModel source patches (bao-fixcode). Fail the build unless hashes match and every patch applies cleanly.
COPY third_party/patches/automodel/ /tmp/vlm/patches/
#    Only patches listed in PATCHES.sha256 are applied, in that order; any unlisted *.patch fails the build.
WORKDIR /tmp/vlm/patches
RUN sha256sum -c PATCHES.sha256 \
 && diff <(awk '{print $2}' PATCHES.sha256 | sort) <(find . -maxdepth 1 -name '*.patch' -printf '%f\n' | sort)
WORKDIR /opt/Automodel
RUN while read -r _ name; do \
      patch -p1 --forward --dry-run < "/tmp/vlm/patches/${name}" && patch -p1 --forward < "/tmp/vlm/patches/${name}"; \
    done < /tmp/vlm/patches/PATCHES.sha256 \
 && rm -rf /tmp/vlm

RUN chmod -R o+rX /opt/Automodel
USER 1000:1000
