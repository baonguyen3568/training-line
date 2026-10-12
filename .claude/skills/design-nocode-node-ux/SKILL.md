---
name: design-nocode-node-ux
description: Use when designing or reviewing anything in the no-code node-graph app for non-technical users — a new node type, port/wire types, template workflow, pre-run validation (preflight) rule, error message copy, Simple/Advanced parameters, or the storage panel. Spec lives in .team/proposals/010-nocode-ux.md.
---

# No-code node-graph UX (bao-design)

Persona: "Lan" — knows her task and data, does not code, does not know LoRA/NVFP4. Every decision is checked against her.

## New node checklist
- [ ] Name = user action in plain words ("Nén model"), 1-sentence everyday description, tech label secondary ("ModelOpt · FP8")
- [ ] Belongs to one lane: ① Dữ liệu ② Chuẩn bị ③ Huấn luyện ④ Chạy thử ⑤ Nén ⑥ Đo điểm ⑦ Xuất
- [ ] Ports use the 4 display families only: Dữ liệu aqua ■ · Model blue (●/◐ adapter/◉ quantized/⬢ export) · Kết quả đo orange (◆/◇) · Dịch vụ ink ▲; value ports muted •.
      Never add a new port color (only 3 hues pass CVD all-pairs) — add a shape variant + label chip instead
- [ ] Simple: 2–4 params with presets (Nhanh / Cân bằng / Chất lượng) + plain explanation; Advanced: full params + YAML/CLI equivalent
- [ ] Resource badge: time range, RAM range, disk (from backend estimator; show "chưa ước lượng được" if none — never invent)
- [ ] Preflight rules it contributes (missing input, data format, RAM vs MemAvailable, disk, license, >20GB download)
- [ ] Error codes it can raise exist in `src/vlm_pipeline/common/error_catalog.yaml` (fixcode codes MEM-/DISK-/DATA-/CFG-…) with ≥1 action; copy keys resolved in `src/vlm_pipeline/i18n/{vi,en}.yaml` (design owns copy; NodeSpec carries keys only)
- [ ] States: ▢ chưa đủ · ○ sẵn sàng · ⧗ chờ · ↺ dùng lại · ◐ chạy x% · ✓ xong · ⏸ tạm dừng · ✕ lỗi · – bỏ qua (color + icon + text)

## Error copy rules (see `.team/knowledge/error-catalog-draft.md` decisions)
3–4 parts: Chuyện gì xảy ra / (Vì sao — only if `why` key exists: concrete runtime cause from params) / Cách sửa [buttons] / ▸ Chi tiết kỹ thuật (collapsed).
"bạn", no blame, title ≤ 8 words stating the consequence, ≤ 2 sentences, absolute GB numbers, buttons state their effect.
Param-only fixes apply directly + diff + Undo; costly/irreversible/structural fixes preview then confirm. No modal chains; toast only for drag-drop feedback.

## Template checklist
Outcome in plain words · time/RAM/disk · difficulty · runs immediately on a small sample dataset (5–10 min smoke, tested by bao-tester) ·
"👉 Thay bằng dữ liệu của bạn" dashed node · opening creates a copy · stored as `templates/*.vlmflow.json`.

## Safety invariants (CLAUDE.md)
No button to stop someone else's job; GPU busy → queue; >20GB download → approval; deletes: dry-run dialog, type-to-confirm >20GB or non-regenerable, protected items locked with reason.

## Review output
Same format as `design-dashboard-review` §4. Test with a non-coder when possible (5 tasks: open template, swap data, fix a wrong wire, read results, free disk).
