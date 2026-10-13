# 014 — B-7 `serve.vllm` = node "Chạy thử" (M1b) · bao-fixcode · 2026-10-13

Mục tiêu: node "Chạy thử" chạy THẬT trên canvas, cho cả 3 template. User đưa 1–4 ảnh + yêu cầu, nhận về câu trả lời
của model. Đầu ra phải giống hệt lúc train (prefix/prompt, tiền xử lý ảnh, decoding) và chạy an toàn trên máy dùng chung.

## Hiện trạng (đã kiểm)
- `vllm/vllm-openai:v0.31.0` (ADR-001, security APPROVED `sha256:3f7dd5b7…`) **chưa có trên máy**. Bản ~22 GB nên phải
  báo PM trước khi pull, rồi probe trên driver 580.142, vì forum báo vLLM 26.04 fail trên 580.173. Image v0.27.1/v0.20
  đang có là của project khác, không dùng (proposal 005).
- Đường HF generate cho Parse đã có golden: remote class + KV cache khớp NVIDIA (`2379059`); prefix và
  generation_config parity nằm trong `scripts/qa/eval_cord_worker.py`.
- Spec `serve.vllm` hiện chỉ có `model → endpoint`. `InferenceParams` chưa có ảnh/yêu cầu (node-params-v1 §5 thì có).
  Chưa có node `playground.chat` nào đọc `Endpoint`.

## Quyết định đề xuất
1. **Job một lần, không phải server sống.** Worker chạy 1 container: load model, chạy các ảnh, ghi `result.json`,
   rồi thoát. Giữ GPU lock + memguard qua `run_job` giống train, nên đúng quy tắc "1 job nặng" (một server sống sẽ
   giữ ~36 GB cho tới khi ai đó tắt). `Endpoint` của v1 trỏ tới `result.json`; endpoint OpenAI sống để cho
   `playground.chat` sau (P009).
2. **Engine:** `engine=hf` luôn có, chạy trong `vlm-automodel` (DEFAULT_TRAIN_IMAGE) bằng
   `AutoModel/AutoModelForImageTextToText` + processor của chính model. `engine=vllm` là `vllm.LLM(...)` offline trong
   image v0.31.0 với `gpu_memory_utilization` ≤0.3, chỉ bật khi image đã pull + probe PASS + có test parity. Trước
   lúc đó `vllm` tự lùi về `hf`, ghi `engine_used` vào kết quả, không âm thầm. Parse luôn dùng `hf` (vLLM ❓ P-4).
3. **Parity với train (gate ADR-001 am. 1):**
   - Parse: prefix = id của task prompt lấy từ đúng collate train (`add_special_tokens=False`, start token ghim)
     → `[2,0,50004,50008,50001]`; assert ở mọi ảnh.
   - Qwen/Cosmos: chat template của processor với `min/max_pixels` lấy từ catalog (cùng nguồn với node Fine-tune).
   - Decoding: `generation_config.json` của snapshot gốc đã pin (Parse: `repetition_penalty` 1.1); chỉ ghi đè
     `max_new_tokens`/`temperature`/`top_p` theo params. Mọi ghi đè được ghi vào `result.json`.
   - Model đã fine-tune: export phải qua `ckpt_complete` (BUG-002), nếu không thì từ chối.
4. **Params mới:** `images` là danh sách 1–4 đường dẫn ảnh trong workspace (SEC-001 kiểm path) và `prompt` (Parse:
   cố định, ẩn). Frontend upload ảnh vào workspace rồi truyền đường dẫn. Cần frontend và design thống nhất widget.
5. **Kết quả:** `result.json` = `{engine_used, model, image, prompt, prefix_ids, decoding, overrides, outputs[{image,
   text, n_tokens, truncated}], worker_sha256}` và `run.json` (peak mem, image digest). UI hiện `text`.

## Kế hoạch
B7-1: params + worker HF + test unit (worker giả container) → B7-2: chạy thật Parse/Qwen trên GPU (golden ảnh NVIDIA cho
Parse) → B7-3 (chờ PM duyệt pull 22 GB): probe vLLM 0.31.0 + parity vLLM↔HF trên Qwen.
Cần: PM (pull vLLM), frontend/design (upload ảnh, hiển thị kết quả), backend (review worker), security (path ảnh).
