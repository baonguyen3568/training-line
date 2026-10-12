#!/usr/bin/env bash
# Local CI (ADR-004 D8). CPU-only, never takes the GPU lock.
#   scripts/check.sh               host gates: lockfile, format, lint, types, import layers, unit tests, secrets
#   scripts/check.sh --container   integration tests inside the NGC image (owner: bao-backend, S-2)
# Stops at the first failing gate. pip-audit is informational only (ADR-003).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "${1:-}" == "--container" ]]; then
  CONTAINER_CHECK="scripts/container/check.sh"
  if [[ ! -x "$CONTAINER_CHECK" ]]; then
    echo "check.sh --container: $CONTAINER_CHECK not implemented yet (owner: bao-backend, S-2)" >&2
    exit 2
  fi
  exec "$CONTAINER_CHECK" "${@:2}"
fi

step "uv lockfile is up to date"
uv sync --locked --quiet

step "ruff format"
uv run ruff format --check .

step "ruff lint"
uv run ruff check .

step "mypy"
uv run mypy
uv run mypy --strict scripts/cleanup

step "API schemas are up to date (schemas/)"
uv run python scripts/export_schemas.py --check

step "import layers (import-linter)"
uv run lint-imports --no-cache

step "unit tests (host)"
uv run pytest -q -m "not gpu and not slow and not eval and not integration"

step "secrets in git history (gitleaks)"
scripts/tools/gitleaks.sh git --redact --no-banner -c .gitleaks.toml

if [[ -f frontend/package.json ]]; then
  step "frontend (biome, tsc, vitest)"
  # `npm run check` = build generated tokens/i18n -> biome ci -> tsc --noEmit -> vitest run (owner: bao-frontend)
  (cd frontend && npm ci --ignore-scripts --silent && npm run check)
fi

step "pip-audit (informational, does not fail the run)"
uv export --locked --no-hashes --no-emit-project --quiet > .cache/requirements-audit.txt
uvx --exclude-newer "7 days" pip-audit --progress-spinner off -r .cache/requirements-audit.txt || true

printf '\n\033[32mAll gates passed.\033[0m\n'
