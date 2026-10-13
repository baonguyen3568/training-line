# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# NTC-Training app (ADR-015, Release 1.0 A3): FastAPI + the built frontend. No docker CLI, no docker socket, no
# secrets in any layer. Jobs go through vlm-runner (VLM_RUNNER_SOCKET). Build context = repo root, filtered by
# docker/app.Dockerfile.dockerignore (allowlist). Base images pinned by digest (docker/IMAGES.lock, both arches).
#   docker buildx build -f docker/app.Dockerfile --build-arg VLM_UID=$(id -u) --build-arg VLM_GID=$(id -g) .

ARG NODE_IMAGE=node:22-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392
ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- frontend (static files only reach the final image)
FROM ${NODE_IMAGE} AS web
WORKDIR /w
COPY frontend/package.json frontend/package-lock.json frontend/
RUN cd frontend && npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ frontend/
COPY design/tokens/ design/tokens/
COPY site/src/icons/lucide/ site/src/icons/lucide/
COPY src/vlm_pipeline/i18n/ src/vlm_pipeline/i18n/
RUN cd frontend && npm run build

# ---------------------------------------------------------------- python deps, locked (uv.lock, hashes)
FROM ${PYTHON_IMAGE} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE}
ARG VLM_UID=1000
ARG VLM_GID=1000
# Never root (security F2): install.sh refuses to run as root, and so does the build.
RUN test "${VLM_UID}" -ne 0 && test "${VLM_GID}" -ne 0
# A passwd entry for the run uid: policy.toml and secrets.key are read from ITS home via pwd (never $HOME, D17).
RUN groupadd --gid "${VLM_GID}" app && useradd --uid "${VLM_UID}" --gid "${VLM_GID}" --home-dir /home/app \
        --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /home/app/.config/ntc-training && chown -R app:app /home/app && chmod 700 /home/app/.config/ntc-training
COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
COPY pyproject.toml ./
COPY configs/ configs/
COPY --from=web /w/frontend/dist/ frontend/dist/
# The commit this image was built from (/api/health, checklist F): `--build-arg VLM_BUILD_COMMIT=$(git rev-parse HEAD)`.
ARG VLM_BUILD_COMMIT=""
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLM_BUILD_COMMIT=${VLM_BUILD_COMMIT} \
    VLM_PROJECT_ROOT=/app VLM_IN_COMPOSE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
USER app
EXPOSE 7870
CMD ["vlm-app", "--worker", "container"]
