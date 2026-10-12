#!/usr/bin/env bash
# Dev launcher for the shared DGX Spark host: tighter plugin policy than an end-user install (P003 D17).
set -euo pipefail
cd "$(dirname "$0")/../.."
export VLM_PROFILE=shared_dev
exec uv run vlm-app "$@"
