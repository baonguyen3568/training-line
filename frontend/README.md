# frontend — NTC Training (node-graph editor)

React 19 + Vite 8 + `@xyflow/react` 12 + zustand (ADR-009). TypeScript strict, Biome, Vitest.
Design: P009 + `009-annex-frontend-canvas.md`, P010 (UX), P003 D13/D16 (security), P004 §8.2 (tooling).

```bash
npm ci --ignore-scripts          # .npmrc: ignore-scripts, save-exact, min-release-age=7
npm run check                    # tokens+i18n → biome ci → tsc → vitest  (used by scripts/check.sh)
VITE_MOCK_API=1 npm run dev      # http://127.0.0.1:4010 with the in-browser mock API (no backend needed)
npm run dev                      # proxies /api → the dev app, VLM_DEV_APP (default 127.0.0.1:4003, D-8)
npm run build                    # dist/, served by FastAPI in production (single origin, no CDN)
npm run licenses                 # dependency licenses for configs/security/licenses.yaml
```

- Ports (D-8, project range 4000–4050): 4010 dev server, 4012 a second (mock) dev server, 4013/4014 E2E, 4011 site.
- Behind nginx under a prefix (e.g. `/app/`) the same build works: assets are relative (`base: "./"`) and every
  API / upload / WebSocket URL goes through `appPrefix()` (src/api/client.ts). nginx must redirect `/app` → `/app/`.
- `src/generated/` is build output (tokens.css from `design/tokens`, i18n JSON from `src/vlm_pipeline/i18n`) — never edit.
- `src/api/types.ts` is hand-written until the backend exports `schemas/`; then replaced by openapi-typescript output.
- `src/api/mock.ts` mirrors the documented API shapes only; validation truth is the backend.
- Untrusted text (third-party nodes, workflow files): plain text only, `src/lib/safeText.ts`; no `dangerouslySetInnerHTML`
  (Biome error), no third-party JS/CSS (P003 D16 T9). Tests: `tests/untrusted-node.test.tsx`.
