# ADR-012 — Export kèm license & ghi công

- Trạng thái: **ACCEPTED** — 2026-10-13
- Nguồn: `.team/proposals/012-export-license-attribution.md` (bao-backend). Review: bao-security, bao-cleancode, bao-research.

## Quyết định
- Mọi export ghi kèm `LICENSE`, `LICENSES/` (văn bản license nguyên văn đã vendor + sha256, security duyệt 1a2ca19), `NOTICE`, `ATTRIBUTION.json` (truy nguồn theo chuỗi run.json), và model card README tối thiểu.
- Fine-tune là **Derivative Model** theo NVIDIA Open Model License: câu notice là bắt buộc. Template Cosmos phải ghi "Built on NVIDIA Cosmos" cả trên site/docs.
- Thiếu thông tin license thì **fail-closed** (EXP-001) ngay ở preflight.
- **Điều kiện ship Export:** bản export phải tự chứa đủ. Code remote (RADIO) được copy kèm hash, `auto_map` được rewrite, và hậu kiểm không còn đường dẫn nội bộ của container (B3, security).
- Thứ tự: commit notices.py → field catalog (cleancode) → notices trong catalog (research) → export worker (backend).
