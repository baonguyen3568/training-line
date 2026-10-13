# BUG-003 — Resume của Nemotron-Parse fail: "Missing key ... input_conditioner.norm_mean"

- **Trạng thái:** CLOSED 2026-10-13 — patch 0005 (feaa1d6) trong `vlm-automodel:26.08-6` (`sha256:dc9da48f…`), kiểm chứng end-to-end bởi bao-backend
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
- **End-to-end trên 26.08-6 (bao-backend):**
  - (a) train mới từ base: `20261013-0924-resume6-a-6da1` completed; log không có dòng "unexpected"/"Missing key"
    (base có 2 key conditioner thừa, DCP bỏ qua mà không cảnh báo).
  - (b) resume từ ckpt của 26.08-5 (`20261013-0849-resume-b1-0335/checkpoints/epoch_0_step_1`, 665 key):
    `20261013-0928-resume6-b2-6b5b` completed, không warning khi load optimizer (FusedAdam).
  - weight_drift A' (liền mạch) vs B2' (resume): decoder.layers lệch 1.41%, ngang nhiễu nền giữa 2 run liền mạch
    (1.37%), rất nhỏ so với thay đổi của 2 step train (17.6%). Loss của B2' lệch A' 2–3e-3, cỡ nhiễu giữa 2 run
    liền mạch (1.7e-3).
- **Việc tiếp theo (backend):** bỏ Parse khỏi `RESUME_UNSUPPORTED`, `DEFAULT_TRAIN_IMAGE = 26.08-6`. Marker
  `.not-resumable` cũ vẫn khoá các run dir đã có marker (an toàn, có chủ đích).
