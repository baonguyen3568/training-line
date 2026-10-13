# M3 D-2 — amd64 + arm64, GPU rời vs GB10 (bao-research, 2026-10-13)

Bằng chứng lấy từ registry chạy không cần pull (`scripts/research/` + tool manifest qua registry API, đọc config blob), PyPI JSON
và docs NVIDIA. Mọi digest dưới đây là digest manifest của từng platform trong index đã pin.

## 1. Image và wheel: có amd64 không?
| Thành phần | Index digest (đã pin) | arm64 | amd64 | Ghi chú |
|---|---|---|---|---|
| `nvcr.io/nvidia/nemo-automodel:26.08` | `sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee` (khớp docker/IMAGES.lock) | `sha256:70b4e9df…b7b55a` (13.9 GB nén) | `sha256:c21c0c3b82f68fcccfded4bf9fe11b2d53df1c9a969e9e883a1562ffca6294f6` (13.9 GB nén) | Cả 2 arch: CUDA 13.3.0.035, `CUDA_DRIVER_VERSION=610.43.02`, PyTorch 2.13.0a0+8145d63 (NGC 26.06) |
| `vllm/vllm-openai:v0.31.0` | `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` | `sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971` (10.1 GB) | `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (9.0 GB) | CUDA 13.0.2; `NVIDIA_REQUIRE_CUDA=cuda>=13.0` |
| `nvidia-modelopt` 0.47.0 (PyPI) | wheel `py3-none-any`, sha256 `db13e62d7efe4a15…` | ✅ | ✅ | Pure Python, không phụ thuộc arch. Image hiện có 0.44.0 (`9b54a853…`, khớp pip freeze) |
| Transformer Engine (FusedAdam) | có sẵn trong image NGC: TE 2.14.1 | ✅ (đã chạy trên GB10) | ✅ (NGC PyTorch build cả 2 arch) | PyPI `transformer_engine_cu13` 2.20.2 có wheel `manylinux_2_28_aarch64` + `x86_64` nếu cần cài ngoài NGC |

`TORCH_CUDA_ARCH_LIST` (SASS được build sẵn):
- automodel arm64: `8.0 8.6 9.0 10.0 11.0 12.0+PTX`.
- automodel amd64: `7.5 8.0 8.6 9.0 10.0 12.0+PTX` (không có 8.9, 11.0).
- vLLM arm64: `8.0 8.7 8.9 9.0 10.0 11.0 12.0`.
- vLLM amd64: `7.5 8.0 8.6 8.9 9.0 10.0 12.0`.
⇒ L40/L40S/RTX 4090 (sm_89) chạy automodel bằng SASS sm_86, tương thích nhị phân trong cùng major 8. GB10 (sm_121) chạy SASS sm_120. Custom kernel JIT (Triton) thì build theo GPU thật.

Các image dẫn xuất của ta (`vlm-automodel:26.08-*`) hiện **chỉ build arm64**. Muốn amd64 thì build lại từ index ở trên (FROM theo index digest, không theo digest arm64), rồi chạy lại gate container trên máy amd64. Security phải review riêng artifact amd64 (digest khác).

## 2. Driver
- Image automodel 26.08 cần CUDA 13.3 ⇒ driver gốc **≥ 610**. Với driver 580 thì cần **forward compatibility** (`cuda-compat-13-3`; bảng compat: 13-3 tương thích với nhánh 580+/590+/595+).
- Forward compatibility **chỉ có** trên "NVIDIA Data Center GPUs", "Select NGC Server Ready SKUs of RTX cards", "Jetson boards" (docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
  - **H100/L40/L40S** (data center): driver 580 cộng compat là chạy được, giống GB10 hiện nay (đã probe PASS trên GB10).
  - **GeForce/RTX tiêu dùng** (4090, 5090…): KHÔNG có forward compat ⇒ cần driver ≥ 610, hoặc chỉ dựa vào minor-version compatibility (không JIT được PTX mới hơn driver). **Phải probe**, không được hứa trước.
- vLLM v0.31.0 (CUDA 13.0) chạy native với driver ≥ 580 trên mọi GPU.
- Quy tắc như hiện nay: **probe từng image trên từng driver/GPU** (G0 env smoke) trước khi đưa vào app.

## 3. Bộ nhớ: UMA (GB10) vs GPU rời
| | GB10 (UMA 128 GB, dùng chung CPU+GPU) | GPU rời (H100 80 GB, L40S 48 GB, RTX 4090 24 GB, RTX 5090 32 GB) |
|---|---|---|
| Ngân sách | Một vùng nhớ: weights + activations + optimizer + RAM host đều trừ vào `MemAvailable` | Hai vùng: **VRAM** (model, activations, optimizer) và **RAM host** (dataloader, process, page cache) |
| Preflight | `MemAvailable ≥ est + dự phòng` (memguard) | Cần thêm kiểm **VRAM trống** (`nvidia-smi --query-gpu=memory.free`); `MemAvailable` vẫn kiểm cho RAM host |
| Catalog `ram_gb_train` | Đo = delta RAM của cả host (Parse 19.8 GB ở P-3b-A) | **Không dùng trực tiếp được.** Ví dụ P-3b-A: `mem_max_alloc_gb` của AutoModel chỉ ~5–6.3 GB, trong khi delta host là 19.8 GB. Cần field riêng `vram_gb_train` (đo bằng `torch.cuda.max_memory_reserved` + overhead CUDA context ~1 GB) và `host_ram_gb_train` |
| `pin_memory` | Để `false` (pinned chỉ nhân đôi RAM trên UMA, 001a §9.1 d) | Nên để `true` (tăng tốc H2D) ⇒ template cần override theo platform |
| Watchdog | Memguard abort khi RAM < 8 GB (UMA có thể làm treo cả máy) | OOM VRAM chỉ giết process (lỗi CUDA OOM), không treo host ⇒ ngưỡng memguard RAM có thể nới |

## 4. Tính năng theo kiến trúc GPU (sm)
| GPU | sm | FP8 (ModelOpt PTQ) | NVFP4 | Ghi chú |
|---|---|---|---|---|
| A100 | 8.0 | ❌ không có FP8 tensor core | ❌ | Catalog `quant_fp8` phải AND với khả năng GPU |
| L40/L40S, RTX 4090 | 8.9 | ✅ | ❌ | |
| H100/H200 | 9.0 | ✅ | ❌ | |
| B200/GB200 | 10.0 | ✅ | ✅ | |
| RTX 5090 / RTX PRO 6000 Blackwell | 12.0 | ✅ | ✅ | |
| GB10 (DGX Spark) | 12.1 | ✅ | ✅ (GPU-003: NVFP4 có thể chậm hơn FP8 trên máy này) | |
⇒ Node "Nén model" cần đọc compute capability lúc preflight (capability < 8.9 ⇒ ẩn FP8; < 10.0 ⇒ ẩn NVFP4).

## 5. Việc cần làm (đề xuất)
1. Catalog: tách `ram_gb_train` thành `host_ram_gb_train` + `vram_gb_train` (đo lại trên GPU rời); giữ `uma_gb_train` = số hiện tại.
2. Preflight/memguard: nhánh "discrete" (VRAM free + RAM host) cạnh nhánh UMA.
3. Image dẫn xuất amd64: build từ index digest; gate G0 + parity trên máy amd64 thật; security review artifact amd64.
4. Template: `pin_memory` theo platform; quant node lọc theo compute capability.
5. Không có máy amd64 + GPU rời để probe trong dự án hiện tại ⇒ mọi điểm trong mục 2–4 là **chưa đo**, chỉ dựa trên tài liệu/manifest.
