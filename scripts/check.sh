#!/usr/bin/env bash
# Local CI (ADR-004 D8). CPU-only, never takes the GPU lock.
#   scripts/check.sh               host gates: lockfile, format, lint, types, import layers, unit tests, secrets
#   scripts/check.sh --container   integration tests inside the NGC image (owner: bao-backend, S-2)
#   scripts/check.sh --clean       host unit tests on a pristine worktree of HEAD (catches uncommitted dependencies)
# Stops at the first failing gate. pip-audit is informational only (ADR-003).
set -euo pipefail

# A stray GIT_DIR / GIT_WORK_TREE would point every git command below at another repository.
unset $(compgen -e GIT_ || true)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "${1:-}" == "--clean" ]]; then
  # Gates on a pristine checkout of HEAD: catches code that depends on uncommitted files (ADR-004 A4).
  # Frontend / site checks run only when HEAD touches them (or CLEAN_FRONTEND=1), so Python-only commits stay ~30 s.
  # Remove clean worktrees left by runs that died (name carries the owner PID; never touch a live run's tree).
  for stale in "$ROOT"/.cache/check-clean.*; do
    [[ -d "$stale" ]] || continue
    owner="${stale##*/check-clean.}"; owner="${owner%%.*}"
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then continue; fi
    if [[ ! "$owner" =~ ^[0-9]+$ ]] && [[ -n "$(find "$stale" -maxdepth 0 -mmin -60)" ]]; then continue; fi
    git worktree remove --force "$stale" >/dev/null 2>&1 || rm -rf --one-file-system "$stale"
  done
  git worktree prune >/dev/null 2>&1 || true

  WORKTREE="$(mktemp -d "$ROOT/.cache/check-clean.$$.XXXXXX")"
  cleanup() {
    git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
    git worktree prune >/dev/null 2>&1 || true
    rm -rf --one-file-system "$WORKTREE" 2>/dev/null || true  # only the mktemp dir created above
  }
  # EXIT alone does not run when bash is killed by a signal (timeout, Ctrl-C): clean up there too.
  trap cleanup EXIT
  trap 'cleanup; exit 130' INT
  trap 'cleanup; exit 143' TERM HUP
  git worktree add --detach --quiet "$WORKTREE" HEAD
  step "clean checkout of $(git rev-parse --short HEAD): unit tests + import layers"
  (cd "$WORKTREE" && VLM_PROJECT_ROOT="$WORKTREE" uv run --quiet lint-imports --no-cache \
    && VLM_PROJECT_ROOT="$WORKTREE" uv run --quiet pytest -q \
       -m "not gpu and not slow and not eval and not integration")
  if [[ "${CLEAN_FRONTEND:-0}" == "1" ]] \
     || ! git diff --quiet HEAD~1 HEAD -- frontend schemas 2>/dev/null; then
    step "clean checkout: frontend (npm ci + npm run check)"
    (cd "$WORKTREE/frontend" && npm ci --ignore-scripts --silent && npm run check)
  fi
  if [[ -f "$WORKTREE/site/package.json" ]] && { [[ "${CLEAN_FRONTEND:-0}" == "1" ]] \
     || ! git diff --quiet HEAD~1 HEAD -- site docs/legal design/tokens frontend/scripts/build-tokens.mjs 2>/dev/null; }; then
    step "clean checkout: site (npm ci + npm run check)"
    (cd "$WORKTREE/site" && npm ci --ignore-scripts --silent && npm run check)
    (cd "$WORKTREE" && uv run --quiet python scripts/qa/check_site_numbers.py --strict >/dev/null)
  fi
  printf '\n\033[32mClean-checkout gates passed.\033[0m\n'
  exit 0
fi

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

step "credential-shaped strings in tracked files (hf_/nvapi-/W&B; PM 2026-10-13)"
uv run --quiet python scripts/security/scan_tokens.py tracked

if [[ -f frontend/package.json ]]; then
  step "frontend (biome, tsc, vitest)"
  # `npm run check` = build generated tokens/i18n -> biome ci -> tsc --noEmit -> vitest run (owner: bao-frontend)
  (cd frontend && npm ci --ignore-scripts --silent && npm run check)
fi

if [[ -f site/package.json ]]; then
  step "site (build, internal links, security gate scripts/security/site_check.py)"
  # Public site (M2-3, owner: bao-frontend); `npm run check` also runs the security gate (placeholders only warn).
  (cd site && npm ci --ignore-scripts --silent && npm run check)
  step "site: every number on a page comes from its listed sources (no made-up numbers, P011 G1)"
  uv run python scripts/qa/check_site_numbers.py --strict >/dev/null \
    || { uv run python scripts/qa/check_site_numbers.py; exit 1; }  # quiet when clean, full report when not
fi

step "pip-audit (informational, does not fail the run)"
uv export --locked --no-hashes --no-emit-project --quiet > .cache/requirements-audit.txt
uvx --exclude-newer "7 days" pip-audit --progress-spinner off -r .cache/requirements-audit.txt || true

printf '\n\033[32mAll gates passed.\033[0m\n'
