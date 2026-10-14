# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# vlm-runner (ADR-015, Release 1.0 A3): the ONLY service with the docker socket. Stdlib server + the project's
# common modules (narrowed mounts, image gate, claims), and the static docker CLI copied from the official image.
# Non-root: the run uid plus the host's docker group (compose group_add). No network (unix socket only).

ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5
ARG DOCKER_CLI_IMAGE=docker:28-cli@sha256:625d9431a9f54c5a2bc90f24f0e1c3d55b1349fd857dd85035f98c2c9acbdd4d

FROM ${UV_IMAGE} AS uv
FROM ${DOCKER_CLI_IMAGE} AS dockercli

FROM ${PYTHON_IMAGE} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE}
ARG VLM_UID=1000
ARG VLM_GID=1000
# Never root (security F2): install.sh refuses to run as root, and so does the build.
RUN test "${VLM_UID}" -ne 0 && test "${VLM_GID}" -ne 0
RUN groupadd --gid "${VLM_GID}" runner && useradd --uid "${VLM_UID}" --gid "${VLM_GID}" --no-create-home \
        --shell /usr/sbin/nologin runner \
    && mkdir -p /var/lib/vlm-runner /run/vlm-runner /run/vlm-egress \
    && chown runner:runner /var/lib/vlm-runner /run/vlm-runner /run/vlm-egress
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=deps /opt/venv /opt/venv
# The commit this image was built from (release report, checklist S3): `--build-arg VLM_BUILD_COMMIT=$(git rev-parse HEAD)`.
ARG VLM_BUILD_COMMIT=""
LABEL org.opencontainers.image.revision="${VLM_BUILD_COMMIT}"
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLM_BUILD_COMMIT=${VLM_BUILD_COMMIT}
USER runner
CMD ["python", "-m", "vlm_pipeline.runner.server"]
