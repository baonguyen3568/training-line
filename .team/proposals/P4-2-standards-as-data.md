# P4-2 — "Chuẩn mực" thành dữ liệu: rule validate + gợi ý cho node

- **Owner:** bao-research (nội dung chuẩn + nguồn) · **Cùng làm:** bao-tester (bộ kiểm dữ liệu), bao-fixcode (mã lỗi/i18n), bao-backend/cleancode (loader + chỗ chạy rule)
- **Trạng thái:** DRAFT để review. Chưa có code.

## 1. Nguyên tắc
1. **Chuẩn là dữ liệu, không là code.**
   - Mỗi chuẩn là một rule trong `configs/standards/*.yaml`, gồm: `id`, `applies_to` (node/param/port), `check` (tên hàm kiểm có sẵn + tham số), `level` (`error` = chặn chạy, `warn` = cảnh báo, `hint` = gợi ý), `message` (key i18n), `source` (URL/ADR/proposal có số).
   - Code chỉ có một bộ hàm kiểm nhỏ, tổng quát (`range`, `ratio_above`, `dup_rate`, `leak_rate`…). Thêm hay sửa chuẩn thì chỉ sửa YAML.
2. **Mọi ngưỡng có nguồn.** Ngưỡng do team tự đặt (không có chuẩn công bố) ghi `source: team-default`, kèm lý do, để sau này hiệu chỉnh bằng số đo.
3. **Không chặn khi không chắc.** Chỉ dùng `error` cho thứ chắc chắn hỏng (sai định dạng, rò test vào train, vượt bộ nhớ). Còn lại là `warn`/`hint` và có nút "Tôi hiểu, vẫn chạy".
4. **Theo model.** Rule nào phụ thuộc model (pixel, độ dài, cách train, FP8) thì đọc giá trị từ catalog, không lặp số trong standards.

## 2. Bốn nhóm chuẩn (bản nháp nội dung)
### 2.1 Định dạng dữ liệu VLM (`data_formats.yaml`)
| Định dạng | Nhận diện | Chuyển về dạng chuẩn nội bộ | Nguồn |
|---|---|---|---|
| HF chat messages (`messages: [{role, content: [{type: image|text}]}]`) | có `messages` | dạng chuẩn nội bộ (không chuyển) | transformers chat templates; AutoModel VLM datasets |
| LLaVA / ShareGPT (`conversations: [{from: human|gpt, value}]`, `<image>` trong text) | có `conversations` | `human→user`, `gpt→assistant`, `<image>` → content image | LLaVA-1.5 / LLaMA-Factory data README |
| Hỏi–đáp phẳng (`image, question, answer`) | 3 cột | 1 lượt user + 1 lượt assistant | MedPix-VQA, CORD (gt_parse) |
| WebDataset / Energon (`.tar` shards: `jpg` + `json`) | file `.tar` | đọc theo shard (cho data lớn) | NVIDIA Megatron-Energon docs; Curator ImageReader |
Rule:
- `error`: không nhận diện được định dạng;
- `error` DATA-001: số ảnh ≠ số chỗ `<image>`;
- `error` DATA-004: ảnh không tồn tại;
- `warn`: có lượt assistant rỗng.

### 2.2 Chất lượng dữ liệu (`data_quality.yaml`)
| Rule | Ngưỡng đề xuất | Level | Nguồn |
|---|---|---|---|
| Ảnh hỏng / CMYK / RGBA / quá nhỏ (DATA-002) | cạnh ngắn < 28 px (= patch 14 × merge 2 của Qwen-VL) | error nếu > 5%, warn nếu > 0 | Qwen2.5/3-VL processor `min_pixels`; catalog |
| Tỉ lệ khung hình cực đoan | > 200:1 | error (processor Qwen từ chối) | Qwen-VL `smart_resize` (ràng buộc aspect ratio < 200) |
| Ảnh lớn hơn trần pixel của model | > `max_pixels` trong catalog | hint ("sẽ bị thu nhỏ") | catalog `max_pixels` |
| Mẫu bị cắt vì quá dài | > 1% mẫu vượt `min_answer_tokens` hoặc seq len | warn | team-default (P002 truncated) |
| Trùng lặp chính xác (sha256 ảnh + text) | > 1% | warn | team-default; Curator exact dedup |
| Gần trùng (pHash Hamming ≤ 4) | > 5% | hint | Curator semantic dedup (eps 0.01) / LLaVA data practice |
| **Rò train↔eval** (G5): pHash ảnh + n-gram 13 câu hỏi so với mọi split eval trong registry | > 0 mẫu | **error** (loại hoặc xác nhận) | P002 B3 G5; GPT-3/Llama contamination practice (13-gram) |
| Dùng split test để train hoặc hiệu chỉnh (calib) | bất kỳ | **error** | ADR-002; `heldout_splits` trong datasets catalog |
| Data quá ít | < 100 mẫu train | warn ("dễ học thuộc") | team-default (O-1: 20 ảnh thì học thuộc) |
| Không có tập kiểm tra | không nối val_data và dataset không có split validation | warn | 001a §10 (mọi quyết định chọn model dựa trên val) |

### 2.3 Recipe train (`train_recipes.yaml`)
| Rule | Chuẩn | Level | Nguồn |
|---|---|---|---|
| LoRA rank | 8–64; alpha ≈ 2× rank | hint ngoài khoảng | LoRA/QLoRA papers; AutoModel recipes (dim 8–16) |
| LR LoRA | 5e-5 – 2e-4 | warn ngoài khoảng | AutoModel / TRL / LLaMA-Factory defaults |
| LR full SFT | 5e-6 – 5e-5 | warn ngoài khoảng | AutoModel Parse (1e-5), Qwen3.5 SFT recipe |
| **Full SFT trên 1 GPU không có master fp32** | optimizer không phải TE FusedAdam `master_weights` | **error** | 001a §10.9 (P-3: chỉ 5.3% trọng số đổi) |
| Số mẫu đã thấy so với recipe upstream | GBS × steps < 25% của recipe gốc | warn ("có thể thiếu train") | 001a §10.8 (P-3 = 15%) |
| Epoch cao trên data nhỏ | > 5 epoch và < 1k mẫu | warn (overfit) | team-default |
| Đóng băng vision tower | mặc định bật; tắt thì hint "chỉ khi data khác miền, lr ×0.1" | hint | research-model-selection skill; AutoModel freeze_config |
| Cách train không hợp model | khác `train_node` trong catalog | error (CFG-010, đã có) | catalog |

### 2.4 Nén (`quantize.yaml`)
| Rule | Chuẩn | Level | Nguồn |
|---|---|---|---|
| Định dạng hợp với model | `quant_fp8` trong catalog; NVFP4 chỉ khi ModelOpt support matrix có (Qwen3-VL/3.5: chỉ FP8) | error nếu unsupported | ModelOpt hf_ptq support matrix |
| Định dạng hợp với GPU | FP8 cần sm ≥ 8.9; NVFP4 cần sm ≥ 10.0 | error | multiarch-gpu.md |
| Dữ liệu calib | 128–512 mẫu, lấy từ train, khác val/test | warn ngoài khoảng; error nếu dùng test | ModelOpt defaults; ADR-002 |
| Nén vision tower | mặc định tắt | hint | ModelOpt "only the language model is quantized" |
| Gate sau nén | Δ benchmark theo `configs/eval/gates.yaml` + sanity top-1 ≥ 0.95, KL ≤ 0.02 | (gate) | P002 B3; probe B-4 |

## 3. Chia việc
- **research:** viết 4 file YAML (nội dung + nguồn). Định nghĩa danh sách hàm kiểm cần có.
- **tester:** hàm kiểm dữ liệu (`dup_rate`, `phash_leak`, `ngram_leak`, `image_stats`, `token_len_stats`) chạy trên dataset artifact. Test với fixture.
- **fixcode:** mã lỗi/i18n mới (DATA-005 leak, DATA-006 dup, TRAIN-001 no-master-fp32, QUANT-001 format-gpu…), theo error_catalog.
- **backend/cleancode:** loader pydantic cho standards (extra=forbid, mỗi rule có `source`); chạy rule ở graph validate (rule tĩnh) và ở node `data.*` (rule cần đọc data). UI hiện `level` + nút "vẫn chạy" với `warn`.

## 4. Câu hỏi mở
1. Leak check đọc toàn bộ split eval trong registry: tốn thời gian (pHash ~ms/ảnh). Đề xuất cache hash của các split eval theo revision.
2. n-gram 13 cho câu hỏi ngắn (VQA thường < 13 token) gần như không bắt được gì ⇒ với câu hỏi ngắn dùng exact match sau chuẩn hoá + pHash ảnh. Cần tester thống nhất.
