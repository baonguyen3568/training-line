---
name: cleancode-check
description: Run the local CI quality gates (ruff format/lint, mypy, import-linter, fast pytest) for the VLM pipeline repo, or set them up / bump tool versions. Use before committing, before review, or when asked "does it pass lint/types/tests".
---

# Local CI gates (no GPU, no network beyond uv cache)

Run from repo root. All commands are CPU-only and must never take the GPU lock.
**After EVERY code commit (ADR-004 A4):** `bash scripts/check.sh --clean` — on a pristine worktree of HEAD: import layers, API schema drift (`export_schemas.py --check`), unit tests (~60 s); plus the frontend gate when HEAD touches `frontend/`, `schemas/` or `configs/templates/`, and the site gate when it touches `site/`, `docs/legal/` or tokens (`CLEAN_FRONTEND=1` forces both). Red = the commit depends on an uncommitted file or left a generated file stale.

```bash
uv sync --locked --group dev                 # tooling env, fails if uv.lock stale
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run lint-imports                          # module boundary contracts (import-linter)
uv run pytest -m "not gpu and not slow and not eval and not integration" -q   # integration runs in NGC container: scripts/check.sh --container
```
Once `scripts/check.sh` exists (Proposal 004), prefer: `bash scripts/check.sh` (same steps, stops at first failure).
Auto-fix (only on files you own, see BOARD/fixcode coordination):
```bash
uv run ruff format <paths> && uv run ruff check --fix <paths>
```

## GPU / slow tests
Only with GPU lock (see CLAUDE.md): check `.team/GPU_LOCK`, write owner/job/time/PID, run `uv run pytest -m gpu`, delete lock.
Never use `-n` (xdist) with gpu tests.

## Bumping tool versions
- Pins are exact (ruff/mypy versions change formatting/diagnostics). Bump ruff in both `pyproject.toml` dev group and `.pre-commit-config.yaml` in one dedicated commit; run `ruff format .` in that same commit.
- Respect the supply-chain cooldown agreed with bao-security (uv `exclude-newer`).
- Check latest: `curl -s https://pypi.org/pypi/<pkg>/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"`

## Triage of failures
- mypy `import-untyped` for NVIDIA libs → add per-module override in `[[tool.mypy.overrides]]`, never global `ignore_missing_imports`.
- import-linter violation → fix the import (move shared code to `common`) rather than adding `ignore_imports`; if unavoidable, add with a comment + TODO owner.
