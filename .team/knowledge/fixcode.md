# Knowledge base — Fix Code / Debugging (bao-fixcode)

> Phạm vi: lỗi thường gặp khi train/quantize/eval/serve VLM bằng stack NVIDIA trên **DGX Spark (GB10, sm_121, aarch64, 128GB UMA, CUDA 13, driver 580.142)**.
> Cập nhật: 2026-10-12. Mức tin cậy: **[OFF]** = docs NVIDIA/PyTorch chính chủ · **[COM]** = báo cáo cộng đồng (forum/blog), cần tự xác minh · **[EXP]** = kinh nghiệm/suy luận, chưa kiểm chứng trên máy này.
> Ai gặp lỗi mới đã fix → thêm 1 mục vào §3 (format: Triệu chứng → Nguyên nhân → Fix → Nguồn/commit).

## 0. Trạng thái máy (đo 2026-10-12 15:58)
- `nvidia-smi`: GB10, driver 580.142, CUDA 13.0, Memory-Usage = **Not Supported** (bình thường trên iGPU).
- `free -g`: total 121, used 78, available ~43, swap 15 (đã dùng 8). → **Ngân sách thực tế cho job của ta ≈ 35–40GB**, không phải 128GB.
- Python host 3.12.3, có `uv`. Docker có sẵn image NGC/NIM của service khác (không đụng vào).

## 1. Nguyên tắc debug (quy trình)
1. **Tái hiện trước, sửa sau.** Ghi lệnh tái hiện tối thiểu + env fingerprint (§4) vào bug report. Không tái hiện được → không đóng bug.
2. **Một biến mỗi lần.** Đổi 1 thứ (version/flag/env), chạy lại, ghi kết quả.
3. **Phân loại lỗi trước khi đoán** (§2): ABI/arch · memory UMA · numerics · data/processor · checkpoint/convert · concurrency.
4. **Root cause, không bịt triệu chứng.** `try/except: pass`, tăng timeout, hạ batch "cho chạy được" chỉ là workaround → ghi rõ là workaround trong commit + mở ticket.
5. **Fix tối thiểu + test chống tái phát** (pytest, CPU-only nếu được; test GPU đánh dấu `@pytest.mark.gpu`).
6. Commit message tiếng Anh: `fix(<module>): <what>` + đoạn `Root cause:` + `Repro:`.
7. Job nặng → chạy qua `run_job`/`jobrun_cli` (preflight + GPU lock `.team/lock/GPU_LOCK`, chỉ xem bằng `gpu_lock status`, không ghi tay). Repro OOM trên máy dùng chung có thể làm **treo cả máy** (§2.2) → dùng mô hình nhỏ/giới hạn bộ nhớ trước.

## 2. Các lớp lỗi đặc thù GB10 / aarch64 / UMA

### 2.1 ABI / kiến trúc (sm_121, CUDA 13, aarch64)
| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `ImportError: libcudart.so.12` / `libcublas.so.12` khi import | Wheel PyPI build cho CUDA 12; Spark chạy CUDA 13 [COM] | Dùng container NGC (`nvcr.io/nvidia/pytorch:<YY.MM>-py3` arm64) hoặc torch từ index `https://download.pytorch.org/whl/cu130`. Kiểm tra `torch.version.cuda` == 13.x |
| `no kernel image is available for execution on the device` | Extension build thiếu arch 12.1 | Rebuild với `TORCH_CUDA_ARCH_LIST="12.1"` (hoặc `12.1a` nếu cần tính năng arch-specific); kiểm tra `torch.cuda.get_arch_list()` có `sm_120`/`sm_121` |
| `FATAL: kernel fmha_cutlassF_..._sm80 is for sm80-sm100, but was built for sm121` | Kernel mem-efficient attention của torch nightly cũ [COM] | Đổi backend SDPA (cuDNN/math), hoặc nâng torch; `attn_implementation="sdpa"` |
| `pip install flash-attn` fail / không có wheel | flash-attn 2/3 whitelist arch, chưa có sm_121 aarch64 wheel [COM] | **Không dùng flash-attn trên bare pip** → `attn_implementation="sdpa"`. NGC container có bản build sẵn. TE FA4 cho sm_121 đang PR #3125 (chưa merge) |
| `ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'` (Triton / torch.compile / vLLM) | Triton dùng ptxas bundled cũ [COM] | `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (ptxas CUDA 13) ; hoặc dùng Triton đi kèm torch chính thức; tạm thời `torch.compile` off |
| NVFP4 chạy **chậm hơn** FP8 (~25-30%) | Không có kernel `sm_121a` native FP4 → fallback Marlin dequant [COM] | Benchmark cả FP8 và NVFP4; không giả định NVFP4 nhanh hơn trên GB10. Báo bao-research/tester |
| Mixed torch/transformers/TE/triton drift sau `pip install X` | Bare pip kéo version khác của torch (x86/CUDA12 hoặc CPU-only!) | Luôn `uv pip install --no-deps` cho package có pin torch; check `pip show torch` sau mỗi install; ưu tiên container NGC |
| `torch.cuda.is_available() == False` trong container | Thiếu `--gpus all` / `--runtime=nvidia`, hoặc torch CPU wheel bị kéo đè | `python -c "import torch;print(torch.__version__, torch.version.cuda)"` — có `+cpu` là bị đè |

**Đo thực tế trên máy này (2026-10-12, bao-fixcode)** — `nvcr.io/nvidia/pytorch:25.11-py3` (torch 2.10.0a0 nv25.11, CUDA 13.0, Triton 3.5.0, cap (12,1)), chạy native trên driver 580.142:
- Image đã **tự set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`** (ptxas CUDA 13.0) → Triton `@jit`, `torch.compile` (Inductor), SDPA bf16 đều OK, không gặp lỗi `sm_121a`.
- ⚠️ **`TRITON_PTXAS_PATH=""` (set nhưng rỗng) làm Triton crash**: `PermissionError: [Errno 13] Permission denied: ''` ở `triton/knobs.py`. Trong compose/docker `-e TRITON_PTXAS_PATH=$VAR` với VAR rỗng là dính lỗi này → chỉ set khi có giá trị, hoặc để image tự lo.
- Probe: `scratchpad triton_probe/probe.py` (add kernel + torch.compile + SDPA), ~1 phút, `--memory=8g`.

### 2.2 Bộ nhớ hợp nhất (UMA) — lớp lỗi nguy hiểm nhất
- **`nvidia-smi` không báo memory** (Not Supported) — đúng thiết kế [OFF]. Đo bằng `/proc/meminfo` (`MemAvailable`, `SwapFree`), `free -g`, `torch.cuda.memory_allocated/reserved`.
- **`cudaMemGetInfo` / `torch.cuda.mem_get_info()` báo THIẾU** vì không tính bộ nhớ có thể reclaim (page cache, swap) [OFF]. Ước lượng đúng: `MemAvailable` (không swap) hoặc `MemAvailable + SwapFree` (có swap). NVIDIA khuyên không dựa riêng vào cudaMemGetInfo [OFF].
- **Page cache ăn chung pool với GPU**: đọc dataset/checkpoint lớn làm phình buffer cache → OOM dù "còn chỗ". Debug-only: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` [OFF] — **cần sudo → phải hỏi PM/user, không tự chạy** (máy dùng chung).
- **Máy treo ("zombie") thay vì ném CUDA OOM**: driver fail cấp phát (`NV_ERR_NO_MEMORY`), `nvidia-modeset` kẹt D-state; `docker --memory=100g` **không** chặn được hoàn toàn [COM]. → Luôn có biên an toàn, theo dõi `MemAvailable` trong lúc chạy, tự dừng job khi < ngưỡng (đề xuất watchdog ở §5).
- **Tranh chấp với service khác**: process khác (kafka/elastic/comfy-ui, NIM...) phình ra có thể đẩy job ta OOM và ngược lại [COM]. Không bao giờ kill process không thuộc dự án → báo PM.
- **Allocator PyTorch**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` giảm OOM do phân mảnh (khuyến nghị đầu tiên trong Megatron-Bridge memory tuning) [OFF]. `torch.cuda.set_per_process_memory_fraction` trên UMA **chưa được xác minh** hành vi [EXP] — test trước khi dựa vào.
- **vLLM `--gpu-memory-utilization`** là tỉ lệ của **cả pool 128GB** dùng chung [OFF-vLLM blog]. Mặc định 0.9 → pre-allocate ~110GB KV → chắc chắn sập máy dùng chung. Với máy này: bắt đầu **≤0.25**, `--max-model-len` vừa đủ, `--max-num-seqs` thấp, `--enforce-eager` tiết kiệm ~13GB (torch.compile + CUDA graphs) [COM].
- **JIT spike**: FlashInfer/Inductor compile kernel lúc khởi động (cicc song song) có thể ngốn thêm 20-30GB tạm thời [COM] → tính vào ngân sách; dùng image có kernel sm_121 build sẵn.
- **Load checkpoint**: safetensors mmap = page cache; NVFP4 load trên sm_121 từng có spike [COM]. `fastsafetensors` + utilization cao → nguy cơ treo [COM].
- **Swap**: máy đang có 8GB swap dùng → hiệu năng tụt mạnh khi chạm swap; coi swap là vùng nguy hiểm, không phải bộ nhớ.

### 2.3 Hiệu năng "kỳ lạ"
- Băng thông thực đo ~180-192 GB/s (spec 273) [COM]; decode bị memory-bound → batch nhỏ/concurrency >4 lợi ích giảm [OFF-vLLM blog].
- Throttle nhiệt/công suất khi chạy dài [COM] → `nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv -l 5`.
- Request đầu tiên chậm (~25s) do JIT → warm-up trước khi benchmark latency [OFF-vLLM blog]. Báo bao-tester để harness loại warm-up.
- Tốc độ chậm bất thường → nghi PTX JIT fallback (kernel không có SASS sm_121). Check `CUDA_CACHE_PATH`, lần chạy thứ 2 có nhanh hơn không.

### 2.4 NCCL / distributed (1 GPU)
- Chỉ 1 GPU → **không cần NCCL thật**; nhiều framework (NeMo/Megatron/torchrun) vẫn init process group. Lỗi hay gặp [EXP]:
  - `NCCL WARN ... no socket interface` / treo lúc init → `NCCL_SOCKET_IFNAME=lo`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT` cố định chưa bị chiếm (máy dùng chung — kiểm tra `ss -ltnp`).
  - Treo khi `/dev/shm` nhỏ trong docker → `--ipc=host` hoặc `--shm-size=16g`. Với DataLoader nhiều worker cũng vậy.
  - Debug: `NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.
  - TP/PP=1, ưu tiên single-process; tránh TP giữa 2 Spark (hang) [COM].

### 2.5 VLM: data / processor / chat template
- `ValueError: Image features and image tokens do not match` → số placeholder ảnh trong text ≠ số feature từ vision encoder. Nguyên nhân thường: chat template không chèn đủ `<image>`/vision tokens; resize ảnh (min/max_pixels) khác giữa lúc build data và lúc train; **truncation cắt mất token ảnh** khi `max_length` nhỏ; lệch version `transformers`/`qwen_vl_utils` [COM].
  → Checklist: đếm token ảnh sau `processor(...)`; không truncate giữa vùng ảnh (lọc mẫu quá dài thay vì cắt); pin version processor giống nhau ở data/train/eval/serve.
- Label masking sai (loss tính cả prompt/image tokens) → loss thấp bất thường nhưng model không học. Test: tỉ lệ `labels != -100` mỗi mẫu.
- `pad_token_id` None / AttributeError khi load Qwen3-VL trên transformers dev 5.0 [COM] → pin `transformers` theo `config.json` của checkpoint (`transformers_version`).
- `trust_remote_code` code của model đổi theo revision → pin `revision=<commit sha>` khi load từ HF.
- Ảnh hỏng/CMYK/RGBA/kích thước 1px → crash worker giữa epoch. Validate ở bước curate, không ở train.

### 2.6 Checkpoint HF ⇄ Megatron (Megatron-Bridge)
- KeyError trên vision tower (vd `model.visual.patch_embed...`) → **prefix key khác nhau** (`model.visual.*` vs `model.language_model.visual.*`) giữa phiên bản transformers; bridge map theo tên literal [COM]. Fix: so `model.safetensors.index.json` với mapping registry của bridge; re-key, không đổi weight.
- Tied embeddings: checkpoint thiếu `lm_head.weight` khi `tie_word_embeddings=true` → converter phải nhân bản/untie đúng cách; Megatron-LM dùng `--untie-embeddings-and-output-weights` cho model untied [OFF]. Kiểm tra config cả 2 phía.
- **Round-trip test bắt buộc** (đề xuất cho mọi converter): HF → Megatron → HF, so sánh từng tensor (`torch.allclose`, atol theo dtype) + so logits trên 1 batch cố định (diff max < 1e-2 bf16). Chỉ "load được" chưa đủ.
- QKV interleave / GQA head ordering sai → load OK nhưng output rác. Phát hiện bằng so logits, không bằng shape.
- dtype: convert ở fp32 hay bf16? Ghi rõ; vision encoder thường nên giữ bf16/fp32.

### 2.6b Lỗi "sai âm thầm" do đổi tên field config (không crash, accuracy tụt)
**CẬP NHẬT 17:05 — root cause thật rộng hơn:** `NemotronParseConfig.__init__` dựng encoder config từ C-RADIOv2-H rồi chỉ
copy key encoder `if hasattr(self.encoder, key)`; `RADIOConfig` không có field normalize → cờ RƠI Ở MỌI VERSION, kể cả
v1.1 (đo trong 26.08-1: flag model thấy = False; tester xác nhận số học: input patch-embed = bf16(((1−m)/s−m)/s)).
Fix: patch `third_party/patches/automodel/0001-…` (image vlm-automodel:26.08-2). Bài học: **đừng tin "bản cũ đúng"
khi chưa đo** — kiểm hành vi model đã dựng, không chỉ tên field trong config.json. Probe gate: so với kỳ vọng làm tròn
theo dtype encoder (bf16), không phải fp32 (lệch ~5e-3).
**Ca ban đầu (bao-research, 2026-10-12):** Nemotron-Parse v1.2/2.0 + class native NeMo AutoModel (26.08 và main@829445d),
`nemo_automodel/components/models/nemotron_parse/model.py` L379 đọc `getattr(config, "image_processor_normalizes", False)`,
nhưng config encoder v1.2/2.0 đổi tên thành `processor_normalizes: true` → `getattr` trả default False → model
**chuẩn hoá ảnh lần 2**. 2.0 còn `tie_word_embeddings: true` vs `TieSupport.UNTIED_ONLY` → từ chối load (lỗi rõ ràng).
v1.1@fd1e3c6b dùng tên cũ → đúng; dự án chọn v1.1.
**Root cause tổng quát:** code đọc config bằng `getattr(cfg, "<tên>", default)` — upstream đổi tên field không alias
⇒ default âm thầm. Mọi `getattr(..., default)` trên config của model là điểm nghi vấn.
**Phát hiện (test, rẻ → đắt):**
1. *Config key coverage* (CPU, không cần weights): với mỗi model family, giữ bảng alias field nhạy cảm
   (`image_processor_normalizes|processor_normalizes`, `tie_word_embeddings`, `rope_scaling|rope_parameters`,
   `max_pixels|max_image_tokens`, `image_mean/std`…). Đọc **mọi alias** ở config gốc + sub-config (vision/text/encoder);
   FAIL nếu các alias mâu thuẫn, WARN nếu có alias mà class native không đọc tên đó.
2. *Double-normalization probe* (CPU, model tiny hoặc chỉ vision tower): ảnh trắng tổng hợp (255) → processor →
   forward hook trên **input của layer đầu vision encoder** (patch-embed). Giá trị kỳ vọng tính chính xác được:
   đã normalize 1 lần ⇒ `(1 - mean_c)/std_c` mỗi kênh (vd ImageNet ≈ 2.25/2.43/2.64); lần 2 ⇒ `((1-m)/s - m)/s`.
   So với kỳ vọng (atol 1e-3) — không dùng mean≈0/std≈1 trên ảnh tự nhiên (không đủ chắc để gate).
3. *Golden logits* (nightly, GPU lock): ảnh + prompt cố định, so logits/top-1 với golden từ đường HF tham chiếu
   (`trust_remote_code` trong sandbox, revision pin). Bắt được cả các rename khác ngoài normalize (R6 của P007).
**Fix khi gặp:** không sửa code NVIDIA tại chỗ — `_nvidia_compat.py` map alias (ghi `WORKAROUND(BUG-NNN)` + link
upstream + version), mở issue upstream; hoặc pin revision model dùng tên cũ (như v1.1).

### 2.7 Quantize (ModelOpt) / export
- VLM: **phải exclude vision tower + projector** khỏi quantize (kiểm tra pattern exclude thực sự match) [COM-NVIDIA skill]. Sau quantize: in danh sách module đã quantize.
- NVFP4 trên GB10 có thể chậm hơn FP8 (§2.1). Calibration dùng ảnh+text thật của domain, không chỉ text.
- Export TRT-LLM/vLLM: lệch version ModelOpt ↔ TRT-LLM ↔ vLLM → checkpoint format không đọc được. Pin bộ ba trong 1 container.

### 2.8 NeMo / AutoModel
- NeMo 26.02 known issue: FP8 trong AutoModel cần `use_linear_ce_loss=False` (hoặc nâng commit) [OFF].
- Playbook DGX Spark có thể lỗi thời → check issues `github.com/NVIDIA/dgx-spark-playbooks` trước khi chạy tốn thời gian [COM].
- vLLM NGC container mới (26.04) từng báo "compatibility mode UNAVAILABLE" với driver 580.x (cần ≥595) [COM] → **chọn tag NGC tương thích driver 580.142**; bao-research xác minh.

## 3. Sổ lỗi dự án (append khi fix bug thật)
| Ngày | Bug | Root cause | Fix / commit | Test |
|---|---|---|---|---|
| 2026-10-12 | Nemotron-Parse (MỌI version, cả v1.1): ảnh normalize 2 lần với AutoModel native; 2.0 bị từ chối vì tied | `NemotronParseConfig` copy key encoder theo `hasattr(RADIOConfig)` → cờ rơi; v1.2+ đổi tên cờ; class `UNTIED_ONLY` | patch `third_party/patches/automodel/0001` → image 26.08-2 | `tests/integration/automodel/test_nemotron_parse_patch.py` (10 FAIL→PASS) + probe tester |
| 2026-10-12 | BUG-001: Parse native `generate()` lặp `<x_0.3906>` mãi | decoder bỏ qua KV cache, generate có cache chỉ đưa token mới nhất ở vị trí 0 | patch 0003 (tắt cache, đưa cả chuỗi) f42f91c | `test_nemotron_parse_generate.py` |
| 2026-10-12 | 0003 v1 làm `generate(**processor(...))` lỗi `model_kwargs are not used: ['attention_mask']` (golden (b) 26.08-4) | override `prepare_inputs_for_generation` khai báo thiếu tham số tường minh của lớp cha; transformers validate kwargs theo **chữ ký** hàm này (+ forward nếu có **kwargs) | 0003 v2 giữ đủ tham số của parent | test so chữ ký với parent + validate đúng output của processor. **Bài học:** override hàm của transformers thì copy nguyên chữ ký parent; test meta phải gọi qua đường validate của generate, không chỉ gọi thẳng hàm |

## 4. Env fingerprint (dán vào mọi bug report)
```bash
uname -m; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
grep -E 'MemTotal|MemAvailable|SwapFree|Cached' /proc/meminfo
python - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("avail", torch.cuda.is_available(), "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("arch_list", torch.cuda.get_arch_list())
for m in ["transformers","accelerate","peft","transformer_engine","megatron.bridge","nemo_automodel","modelopt","vllm","triton","flash_attn"]:
    try: mod=importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m, "N/A", type(e).__name__)
EOF
# env vars: dùng allowlist trong script dưới (KHÔNG dùng `env | grep` — rò proxy credential/token)
```
(Script gói sẵn: skill `fixcode-triage`.)

## 5. Đề xuất chủ động (sẽ đưa vào proposal)
1. **Memory watchdog** cho mọi job nặng: đọc `/proc/meminfo` mỗi 5s, nếu `MemAvailable < 8GB` → checkpoint + dừng có kiểm soát (tránh treo máy chung).
2. **Preflight** trước train: check arch, torch.cuda 13.x, memory budget, GPU_LOCK, `/dev/shm`.
3. **Round-trip test** bắt buộc cho converter HF⇄Megatron và quantize export.
4. Pin bộ version theo 1 container NGC (giao bao-research chốt tag).

## Nguồn
- NVIDIA DGX Spark Known Issues: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
- NVIDIA DGX Spark Porting Guide — Optimization: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html
- DGX Spark VLM fine-tuning playbook + troubleshooting: https://build.nvidia.com/spark/vlm-finetuning , https://build.nvidia.com/spark/vlm-finetuning/troubleshooting.md
- vLLM on DGX Spark (2026-06-01): https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- Forum — Memory creep on DGX Spark: https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
- Forum — Spark zombie instead of CUDA OOM: https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752
- Forum — sm_121a ptxas error: https://forums.developer.nvidia.com/t/running-llama-3-1-8b-fp4-get-triton-error-value-sm-121a-is-not-defined-for-option-gpu-name/348808
- Forum — SM121 software support roadmap: https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663
- Forum — vLLM 26.04 driver compat: https://forums.developer.nvidia.com/t/vllm-26-04-py3-compatibility-mode-is-unavailable-on-driver-580-173-02-no-dgx-dashboard-path-to-595-58/379168
- PyTorch forum — GB10 CUDA 13 sm_121: https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
- Flash Attention on sm_121: https://medium.com/@rakshith.d26/flash-attention-on-sm-121-solving-pytorch-compatibility-on-blackwell-gb10-a83d9ff3cf9b
- TransformerEngine PR #3125 (SM_121 FA4)
- Spark training gotchas (community skill): https://tessl.io/registry/skills/github/wshobson/agents/spark-training-gotchas
- NVFP4 vs FP8 on GB10: https://ai-muninn.com/en/blog/dgx-spark-nvfp4-trap-gb10-fp8-wins
- NeMo Framework 26.02 known issues: https://docs.nvidia.com/nemo-framework/user-guide/26.02/knownissues.html
- Megatron-Bridge memory tuning: https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/perf-techniques/memory-tuning/SKILL.html
- Megatron-LM Llama/Mistral (untie embeddings): https://docs.nvidia.com/megatron-core/developer-guide/nightly/llama_mistral.html
- HF forum — Image features and image tokens do not match: https://discuss.huggingface.co/t/valueerror-image-features-and-image-tokens-do-not-match/150302
- transformers issue #43334 (Qwen3-VL on 5.0.dev): https://github.com/huggingface/transformers/issues/43334

## 2.2b CUDA OOM lúc nạp model khi host còn nhiều RAM (GB10, quan sát 2026-10-13, CHƯA rõ cơ chế)
- Triệu chứng: `torch.AcceleratorError: CUDA error: out of memory` ở `model.to("cuda")` hoặc ngay lúc tạo CUDA context,
  với MemFree 25–35 GB và MemAvailable 45–50 GB. Peak của job chỉ 2.4 GB. Từng lặp 3/3 lần trong khoảng 90 s, rồi tự hết.
  Runs: `20261013-2002-bench-r0-*`, `20261013-2011/2012-e1val-e1-s599-*`.
- Đã đo được: `cudaMemGetInfo().free` ≈ **MemFree**, không phải MemAvailable, vì page cache không được tính là trống
  cho CUDA. Ví dụ: free 12.9 GiB, MemFree 13.7, MemAvailable 34.
- Đã LOẠI bằng thí nghiệm có kiểm soát (2026-10-13 20:25–20:30, 25/25 lần OK):
  - checkpoint khác nhau: hai checkpoint giống hệt;
  - container cgroup `--memory`: CUDA KHÔNG bị tính vào cgroup, 6 GiB qua được với `--memory 4g`;
  - driver chậm trả bộ nhớ sau khi container 20–30 GiB thoát: MemFree hồi ngay;
  - page cache sạch từ việc đọc checkpoint 3.7 GB chặn cudaMalloc;
  - dirty/writeback: job trước là eval, không ghi checkpoint.
  - container trước dùng /dev/shm lớn (7 GiB, --shm-size 8g), hoặc bị SIGKILL: Shmem và MemFree hồi trong ≤ 2 s, B 16/16 OK
    (20:48–20:51).
- Đang có: 2 VLLM::EngineCore ngoài dự án (khoảng 58 GB) và stack vss-* chạy song song (đã báo PM để hỏi user).
- Biện pháp hiện tại:
  - log CUDA OOM → MEM-003 (d83e0db);
  - executor retry 1 lần sau 120 s nếu OOM lúc khởi tạo: không có step, peak < 50% est, ≤ 300 s, không bị oom_killed
    (backend, `retried_after: gpu_alloc_transient`).
- Lần sau gặp thì ghi lại:
  - giờ chính xác lúc chết;
  - `docker ps -a` của container trước, xem nó đã thật sự biến mất chưa;
  - `nvidia-smi --query-compute-apps`;
  - đối chiếu với sampler 2 s ở `outputs/debug/cuda-oom-20261013/gpu_mem_2s.log`.
