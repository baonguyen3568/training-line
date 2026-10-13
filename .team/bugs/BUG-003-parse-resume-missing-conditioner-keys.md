# BUG-003 — Resume của Nemotron-Parse fail: "Missing key ... input_conditioner.norm_mean"

- **Trạng thái:** FIX PROPOSED (patch 0005, chờ security + cleancode + backend review → rebuild image → resume smoke)
- **Owner:** bao-fixcode · **Báo bởi:** bao-backend (FusedAdam resume smoke, 2026-10-13)
- **Ảnh hưởng:** mọi lần resume Parse v1.1 (nút "Chạy tiếp", hoặc chạy lại vào run dir còn checkpoint). Tạm thời
  backend đưa Parse vào `RESUME_UNSUPPORTED` (cất ckpt sang `*.not-resumable-*`, marker `.not-resumable`, train lại từ 0).

## Tái hiện
- Run cha: `outputs/runs/20261013-0849-resume-b1-0335/checkpoints/epoch_0_step_1` (26.08-5, FusedAdam recipe).
- Resume: `outputs/runs/20261013-0851-resume-b2-a93e/logs/stdout.log` dòng 221–274:
  `RuntimeError: Missing key in checkpoint state_dict: encoder.model_encoder.radio_model.input_conditioner.norm_mean.`
  (`checkpointing.load_model` → `dcp.load`, strict).
- CPU/meta, không cần GPU (scratch `check0005.py`): model mới dựng 667 key, shard 665 key; thiếu đúng
  `input_conditioner.norm_mean/norm_std`.

## Root cause
`RadioWithNeck._externalize_radio_input_conditioner()` (gọi `make_preprocessor_external()` → thay conditioner bằng
`nn.Identity`) chỉ chạy ở **forward đầu tiên**. Checkpoint lưu sau step ≥1 nên không có 2 buffer; model dựng để resume
chưa forward nên vẫn có → DCP strict đòi key không tồn tại. Cùng họ với BUG-002 (state_dict "sống" thay đổi sau forward).

## Fix (patch 0005)
Gọi `_externalize_radio_input_conditioner()` ngay trong `RadioWithNeck.__init__` (lời gọi ở forward giữ nguyên, idempotent).
- state_dict giống nhau trước/sau forward → resume khớp; ckpt cũ của 26.08-5 (665 key) cũng resume được.
- Load base: DCP chỉ đòi key của model, key thừa trong file base bị bỏ qua (`allow_checkpoint_key_subset` không bật).
- Toán không đổi: processor đã chuẩn hoá ảnh, conditioner là Identity từ step 1.
- Export HF vẫn cần `ckpt_complete` (BUG-002), vì class remote của NVIDIA cần 2 buffer.

## Kiểm chứng
- Patch 0001–0005 apply sạch trên base 922e83e4 (`patch --forward --dry-run`).
- `tests/integration/automodel/test_nemotron_parse_resume.py`: 26.08-5 → 1 FAIL (đúng bug), patched → 2 PASS
  (kèm test chiều ngược: `image_processor_normalizes=false` thì conditioner phải được giữ).
- Key của model patched == key của shard thật (665 == 665).
- **Còn thiếu:** resume smoke trên GPU với image mới (backend), sau đó bỏ Parse khỏi `RESUME_UNSUPPORTED`.
