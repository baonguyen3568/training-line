---
name: frontend-dashboard-dev
description: Use when starting, developing, running or debugging the project dashboard (frontend/) on the shared DGX Spark — port allocation, localhost binding, GPU-lock/memory awareness, arm64 deps, and the pre-merge checklist for UI changes.
---
# Dashboard dev workflow (DGX Spark, shared host)

## 1. Before running anything
```bash
# Free ports? Project range is 7870-7879 (7860/6006/8080/11000 are taken by other services)
ss -ltn | awk '{print $4}' | grep -E ':(787[0-9])$' || echo "7870-7879 free"
free -g | head -2                      # dashboard must stay < 500MB RSS
cat .team/GPU_LOCK 2>/dev/null || echo "no GPU lock"
```
- Dashboard NEVER takes the GPU lock and NEVER loads a model (no `import torch` in frontend code).
- Port map (agreed with bao-backend): 7870 dashboard, 7871 vLLM serve, 7872 TRT-LLM serve, 7873 dashboard API (if split), 7874 Trackio/TensorBoard (if adopted). Update `.team/knowledge/frontend.md` if it changes.

## 2. Binding & exposure rules
- Always bind `127.0.0.1` (Gradio: `server_name="127.0.0.1"`, never `share=True`; uvicorn `--host 127.0.0.1`; Next `-H 127.0.0.1`).
- Remote access: `ssh -L 7870:127.0.0.1:7870 <host>`. Any other exposure → ask bao-security + bao-pm first.
- Container (if any) name prefix `vlm-`, arm64 image, `-p 127.0.0.1:7870:7870`.

- Gradio `/demo`: `analytics_enabled=False`, `share=False`, `max_file_size` set, no CDN fonts/JS; capture real headers (`curl -sI http://127.0.0.1:7870/demo/`) and send to bao-security before merge. No 'unsafe-eval' unless proven necessary.

## 3. Data access
- Read only from `outputs/runs/<run_id>/` and `benchmarks/` via the shared pydantic schema (see skill `frontend-metrics-contract`).
- Tail JSONL incrementally (remember byte offset); downsample server-side (LTTB / min-max buckets) to <= ~2000 points per series.
- Missing/`null` metrics (e.g. GB10 "Not Supported" GPU memory) render as "n/a", never 0, never crash.

## 4. Inference demo
- Client only: call the running serve endpoint (OpenAI-compatible `/v1/chat/completions`, `image_url` / `video_url` parts).
- If `.team/GPU_LOCK` exists and owner != serve job → show warning banner (perf numbers would be polluted).
- Limit uploads (image <= 20MB, video <= 200MB / 60s); sanitize model output markdown (no raw HTML).

## 5. arm64 deps
- Python: pin versions in pyproject (`uv add gradio==X.Y.Z`), `uv lock`.
- Node: `npm ci` with committed lockfile; verify native deps have linux-arm64 prebuilt: `npm ls esbuild @swc/core sharp`.

## 6. Pre-merge checklist (UI change)
- [ ] `uv run pytest tests/frontend -q` passes (schema parsing + fixture run renders)
- [ ] Starts with fixture data in `tests/fixtures/runs/` without any GPU/serve running
- [ ] RSS check: `ps -o rss= -p <pid>` < 500MB after loading the largest fixture
- [ ] Bound to 127.0.0.1 only (`ss -ltnp | grep 787`)
- [ ] Matches bao-design tokens (colors/typography), dark + light
- [ ] Reviewed by at least one other member (bao-design for UI, bao-backend for API/schema)
