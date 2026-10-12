# frontend — VLM Studio (node-graph editor)

React 19 + Vite 8 + `@xyflow/react` 12 + zustand (ADR-009). TypeScript strict, Biome, Vitest.
Design: P009 + `009-annex-frontend-canvas.md`, P010 (UX), P003 D13/D16 (security), P004 §8.2 (tooling).

```bash
npm ci --ignore-scripts          # .npmrc: ignore-scripts, save-exact, min-release-age=7
npm run check                    # tokens+i18n → biome ci → tsc → vitest  (used by scripts/check.sh)
VITE_MOCK_API=1 npm run dev      # http://127.0.0.1:7875 with the in-browser mock API (no backend needed)
npm run dev                      # proxies /api → FastAPI at 127.0.0.1:7870
npm run build                    # dist/, served by FastAPI in production (single origin, no CDN)
npm run licenses                 # dependency licenses for configs/security/licenses.yaml
```

- `src/generated/` is build output (tokens.css from `design/tokens`, i18n JSON from `src/vlm_pipeline/i18n`) — never edit.
- `src/api/types.ts` is hand-written until the backend exports `schemas/`; then replaced by openapi-typescript output.
- `src/api/mock.ts` mirrors the documented API shapes only; validation truth is the backend.
- Untrusted text (third-party nodes, workflow files): plain text only, `src/lib/safeText.ts`; no `dangerouslySetInnerHTML`
  (Biome error), no third-party JS/CSS (P003 D16 T9). Tests: `tests/untrusted-node.test.tsx`.
