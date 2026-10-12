# Knowledge — Backend / ML Engineering (bao-backend)
_Cập nhật: 2026-10-12. Nguồn ghi cuối mỗi mục. Đánh dấu: ✅ đã kiểm trên máy này · 📄 theo docs chính chủ · 🗣 community (cần verify)._

## 0. Hiện trạng máy (✅ đo 2026-10-12 16:xx)
- GB10, driver **580.142**, aarch64, Python host 3.12.3, `uv` có sẵn, **không có `gh`**.
- RAM: total 121G, **used ~77G, available ~44G**, swap 8/15G đã dùng (service khác: vLLM voice-agent, open-webui, ...).
  ⇒ **Ngân sách thực cho job dự án ≈ 35–40 GB** trừ khi PM/user tắt service khác.
- Image có sẵn (KHÔNG thuộc dự án, không xoá): `nvcr.io/nvidia/pytorch:25.11-py3`, `nvcr.io/nvidia/vllm:25.12.post1-py3`,
  `vllm/vllm-openai:v0.27.1`, nhiều NIM. Đang pull: `nvcr.io/nvidia/nemo-automodel:26.08`.
- `nvidia-smi` báo Memory-Usage "Not Supported" trên GB10 (bình thường với iGPU) → đo memory bằng `/proc/meminfo` + `torch.cuda.max_memory_allocated()`.

## 1. Phiên bản upstream mới nhất (✅ GitHub API 2026-10-12)
| Thành phần | Release mới nhất | Ghi chú |
|---|---|---|
| NeMo AutoModel | v0.6.0 (2026-08-26) | container `nemo-automodel:26.08` ≈ nhánh v0.6.0 |
| Megatron-Bridge | v0.6.2 (2026-09-18) | |
| NeMo-RL | v0.7.0 (2026-07-29) | chưa thấy recipe VLM GRPO chính chủ → cần research xác minh |
| ModelOpt | 0.47.0 (2026-09-23) | VLM PTQ đã gộp vào `examples/hf_ptq` |
| TensorRT-LLM | v1.2.1 (2026-04-20) | release container multi-arch có arm64 🗣 |
| vLLM | v0.31.0 (2026-10-05) | |

## ⚠️ Ràng buộc driver (user chốt 2026-10-12)
Giữ driver 580 → image mặc định phải CUDA ≤ 13.1 (580 chỉ forward-compat tới 13.1 theo research; 13.2+ báo không chạy). automodel 26.08 (13.3) chỉ là CANDIDATE. Mặc định dự kiến: automodel 26.02 (CUDA 13.0), pytorch 25.11, vllm 25.12.post1/26.01, trtllm 1.2.1 (13.1).

## 2. Train — NeMo AutoModel (khuyến nghị mặc định, HF-native)
- CLI mới: `automodel <config.yaml> [--nproc-per-node 1]` (chạy qua `examples/.../finetune.py` đã **deprecated**). Override kiểu
  `--step_scheduler.max_steps 20`. 📄
- Recipe VLM: `recipe: FinetuneRecipeForVLM`, model `nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained`,
  processor `transformers.AutoProcessor`, loss `MaskedCrossEntropy`, dataloader `torchdata StatefulDataLoader`,
  `freeze_config` (freeze_vision_tower / freeze_language_model), checkpoint `safetensors` + `save_consolidated: true` → ra HF ckpt dùng thẳng. 📄
- Họ model có recipe VLM (`examples/vlm_finetune/`): qwen2_5, qwen3, qwen3_5 (0.8B/2B/4B/9B/27B), qwen3_8, gemma3/3n/4,
  internvl, llava_onevision, phi4, mistral*, kimi, nemotron, nemotron_omni, nemotron_3_5_super_vl… 📄
  - Recipe nhỏ nhất để spike: `qwen3_5/qwen3_5_4b.yaml` (đổi model id sang `Qwen/Qwen3.5-2B` / `0.8B`), dataset MedPix-VQA, `attn: sdpa`.
- Recipe "_spark" cho LLM: `qwen3_8b_squad_spark.yaml`, `llama3_3 ... qlora_rtx_spark.yaml`... chứa setting GB10 (force_hf, attn). 📄🗣
- Số đo memory trên GB10 (Classmethod 2026, LLM nhưng là mốc tham khảo) 🗣: Qwen3-8B LoRA BF16 ≈ **20.8 GiB**, Qwen3-14B LoRA ≈ 43 GiB,
  30B-A3B LoRA ≈ 63 GiB, Qwen3-8B **full SFT ≈ 78 GiB** (72 s/step). LoRA 8B ≈ 1.07k tok/s.
  ⇒ Với ~40 GB: **LoRA ≤ 8B** an toàn; full SFT chỉ cho model ≤ 2–4B.
- Pitfalls 🗣📄:
  - Container chạy root → checkpoint root-owned; chạy với `--user $(id -u):$(id -g)` hoặc chown trong project.
  - Loss/mem/tps in ra **stderr** — đừng tách `2>` mất log; tốt hơn tự ghi JSONL (xem §6).
  - `nemo_automodel.__version__` có thể lệch tag container → ghi lại cả image digest + version trong metadata run.
  - Playbook build.nvidia.com có lúc tụt hậu tag (26.02) → luôn tự kiểm tra tag mới nhất.
  - Docker flags chuẩn: `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g`.
  - pip wheel AutoModel là pure-Python; TE/bitsandbytes aarch64 phải có sẵn → **dùng container**, không tự build trên host.
- Megatron-Bridge: chỉ dùng khi cần Nemotron VL lớn / TP-PP; 1 GPU không có lợi, thêm bước convert HF↔Megatron.

## 3. RL — NeMo-RL
- v0.7.0; GRPO/DPO, dùng vLLM làm generation backend (Ray). Trên 1 GPU UMA: policy + vLLM colocated → memory là nút thắt; nên
  bắt đầu từ VLM ≤ 3B. Chưa xác minh recipe VLM chính chủ → để Phase 2 (P2-4) sau khi SFT chạy ổn. 🗣
- Playbook VLM Spark (Qwen2.5-VL-7B GRPO wildfire, InternVL3-8B video) yêu cầu **không có process GPU khác** → xung đột với máy dùng chung. 📄

## 4. Quantize — ModelOpt (`nvidia-modelopt` 0.47)
- Entry: `examples/hf_ptq/hf_ptq.py --pyt_ckpt_path <hf> --qformat <fmt> --kv_cache_qformat <fp8_cast|nvfp4|none> --export_path <out>`,
  hoặc `--recipe <path>` (ưu tiên hơn qformat). Shell: `scripts/huggingface_example.sh --model X --quant fp8 --vlm`. 📄
- Export **Unified HF checkpoint** (`modelopt.torch.export.export_hf_checkpoint`) → deploy TRT-LLM (PyTorch backend) / vLLM / SGLang. 📄
- VLM: mặc định **chỉ quantize LLM**, vision encoder giữ BF16. 📄
- **Support matrix VLM (quan trọng cho chọn model)** 📄:
  | Model | fp8 | int4_awq | nvfp4 |
  |---|---|---|---|
  | Qwen2/2.5-VL | ✅ | ✅ | ✅ |
  | Qwen3-VL / Qwen3.5 | ✅ (+ FP8 vision recipe) | ❌ | ❌ |
  | Gemma 3 | ✅ | ❌ | ❌ |
  | Nemotron VL | ✅ | ❌ | ✅ |
  | Llava | ✅ | ✅ | ❌ |
  ⇒ Nếu yêu cầu đủ FP8 + NVFP4 + AWQ thì chỉ Qwen2.5-VL đạt; Qwen3.5 mới hơn nhưng chỉ FP8 (validated). Cần research cân nhắc.
- Qwen3-VL/3.5 FP8 vision: `--recipe model_type/qwen3_5/ptq/fp8_vision_lm-kv_fp8_cast --calib_with_images --calib_size 512 --skip_generate`.
- NVFP4 accuracy tốt hơn với `nvfp4_mlp_only` / `nvfp4_omlp_only` (không quantize QKV). Model nhỏ mất accuracy nhiều hơn → cân nhắc QAT.
- VILA/NVILA đã bị bỏ hỗ trợ (0.46).

## 5. Serve / perf
- **vLLM** (khuyến nghị cho đo latency/throughput đầu tiên): image pin theo tag/digest, `--gpu-memory-utilization` phải tính theo
  **ngân sách thực** (máy dùng chung: ≤0.30, KHÔNG để mặc định 0.9 — docker `--memory` không chặn được CUDA alloc trên UMA 🗣),
  `--max-num-seqs` thấp (≤4), `--max-model-len` vừa đủ. 📄 vLLM blog 2026-06.
- **NVFP4 trên GB10 chưa chắc nhanh hơn FP8**: đo Qwen3.6-35B-A3B vLLM 0.19: FP8 53.8 tok/s vs NVFP4 40.8 tok/s (Marlin fallback);
  nguyên nhân do chọn kernel phần mềm (sm_121a), không phải silicon 🗣 → benchmark phải đo **cả FP8 và NVFP4**, không giả định.
- Bandwidth GB10 273 GB/s spec (~180–190 thực) → decode dense bị bandwidth-bound; batch nhỏ.
- **TRT-LLM**: container `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (arm64), `trtllm-serve <ckpt> --backend pytorch`, playbook dùng port 8355 (LLM) / 8356 (VLM). Chưa có ví dụ VLM chính chủ trên Spark → spike sau.

## 6. Pitfalls GB10 unified memory (bắt buộc nhớ)
1. `cudaMemGetInfo` / `torch.cuda.mem_get_info()` **báo thiếu** (không tính page cache/swap reclaimable) → guard bằng `MemAvailable` từ `/proc/meminfo`. 📄
2. Docker `--memory` **không giới hạn** allocation CUDA → phải tự giới hạn (batch/seq len, `gpu_memory_utilization`, `PYTORCH_CUDA_ALLOC_CONF`). 🗣
3. Page cache khi load safetensors cũng ăn chung pool → load model lớn có thể OOM lúc khởi động. `drop_caches` cần sudo → **không làm**, báo PM. 📄
4. Wheel CUDA 12 trên host CUDA 13 → segfault/import error. Dùng container NGC hoặc wheel `cu130`. flash-attn khó build → dùng **SDPA**. 🗣
5. Swap đang dùng 8G → OOM killer có thể giết process của người khác — **job dự án phải fail sớm** thay vì đẩy máy vào swap.

## 7. Thiết kế backend dự kiến (để đưa vào proposal sau ADR-001)
- `src/vlm_pipeline/common/`: `config.py` (pydantic v2, load YAML + override CLI), `gpu_lock.py` (file lock `.team/GPU_LOCK`,
  `fcntl.flock` + nội dung JSON who/job/start/pid, kiểm PID còn sống, KHÔNG tự xoá lock của người khác), `memguard.py`
  (đọc MemAvailable, refuse khi < ngưỡng, watchdog trong lúc chạy), `metrics.py` (JSONL writer), `runinfo.py` (git sha, image digest, versions, seed).
- CLI: `vlm data|train|quantize|eval|serve <config.yaml> [--set k=v]` (typer). Mỗi bước wrap tool chính chủ (gọi `automodel`, `hf_ptq.py`, `vllm serve`)
  trong container `vlm-*`, không re-implement training loop.
- Metrics: theo contract của bao-frontend (`.claude/skills/frontend-metrics-contract/SKILL.md`): wide-format JSONL, `schema_version`
  mỗi dòng, đơn vị trong tên key (`_gb`, `_ms`), append+flush. Backend thêm vào run.json: `stage`, `parent_run_id`,
  `container_image`, `container_digest`. W&B/MLflow = sink phụ, mặc định tắt.
- Output layout (đề xuất gộp frontend + cleanup): `outputs/runs/<run_id>/{run.json,metrics.jsonl,system.jsonl,events.jsonl,logs/,checkpoints/,export/,eval/}`,
  run_id = `<yyyymmdd-hhmm>-<recipe>-<model>-<hash4>`. Cache: `HF_HOME=$ROOT/.cache/hf`.
- Ports dự án (bind 127.0.0.1): 7870 dashboard, 7871 vLLM, 7872 TRT-LLM.

## 8. Checklist trước mỗi job nặng
- [ ] `.team/GPU_LOCK` trống (hoặc lock cũ có PID chết → báo PM, không tự xoá lock người khác)
- [ ] `MemAvailable` ≥ ngân sách job + 8 GB headroom
- [ ] image pin bằng digest; container tên `vlm-*`; `--user` không phải root
- [ ] seed, config, git sha ghi vào run.json
- [ ] HF token qua `.env`, không in ra log
- [ ] model/dataset > 20 GB → báo PM trước
- [ ] xong: xoá GPU_LOCK, container `--rm`

## Nguồn
- NeMo fine-tune playbook: https://build.nvidia.com/spark/nemo-fine-tune (instructions.md, cập nhật 09/2026)
- VLM fine-tune playbook: https://build.nvidia.com/spark/vlm-finetuning
- AutoModel repo + recipe: https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune
- Classmethod AutoModel trên DGX Spark: https://dev.classmethod.jp/en/articles/nvidia-nemo-automodel-dgx-spark-first-touch/
- DGX Spark known issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- ModelOpt hf_ptq README: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq ; changelog https://nvidia.github.io/Model-Optimizer/
- vLLM on DGX Spark (2026-06): https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- NVFP4 vs FP8 GB10: https://ai-muninn.com/en/blog/dgx-spark-nvfp4-trap-gb10-fp8-wins
- UMA container memory runaway: https://forums.developer.nvidia.com/t/urgent-dgx-spark-gb10-uma-nim-container-memory-runaway-all-limitation-parameters-invalid-memory-occupies-120gb/371180/1
- Spark training gotchas: https://github.com/natolambert/dgx-spark-setup
- TRT-LLM Spark playbook: https://build.nvidia.com/spark/trt-llm
