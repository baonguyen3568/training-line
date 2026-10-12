# Proposal 008 — Backend runtime: host orchestrator ↔ container worker, job runner, run manifest
Tác giả: bao-backend · Trạng thái: DRAFT (chờ số đo spike P1-9 + ADR-001) · Ngày: 2026-10-12
Liên quan: P001 (stack), P004 (layout/CI), P005 (storage), P006 (metrics contract), P007 (memguard/preflight).

## 1. Vấn đề
Mỗi bước pipeline phải chạy bằng 1 lệnh, reproducible, an toàn trên host chung (GB10 UMA, ~40GB khả dụng), nhưng
tooling NVIDIA (AutoModel, ModelOpt, vLLM/TRT-LLM) chỉ chạy ổn trong container NGC arm64 — không cài được lên host venv.

## 2. Đề xuất
### 2.1 Hai tầng
| Tầng | Chạy ở | Deps | Trách nhiệm |
|---|---|---|---|
| `vlm` CLI (host) | uv venv host | pydantic, pyyaml, typer (KHÔNG torch) | parse + resolve config, preflight (P007), GPU_LOCK, tạo run dir + run.json, `docker run` worker, tail log → metrics.jsonl, cập nhật status |
| `worker` (container) | NGC image pin @sha256 | chỉ dep có sẵn trong image | gọi API chính chủ (`automodel`, `hf_ptq.py`, `vllm serve`), ghi artifact vào run dir |
Giao tiếp: file `run_dir/resolved_config.json` (in) + `run_dir/{metrics,events}.jsonl`, `export/` (out). Không RPC.

### 2.2 Job runner (`common/jobrun.py`)
1. `preflight(est_peak_gb)` (memguard của P007) → refuse nếu thiếu RAM/swap/disk.
2. Acquire `.team/GPU_LOCK` atomic: ghi JSON {who,job,run_id,host_pid,started_at,token} ra file tạm rồi `os.link(tmp, lock)` (fail nếu đã tồn tại → không có khoảng file rỗng); `LockInfo` = frozen dataclass. Lock của người khác: không bao giờ xoá; PID chết → báo PM.
   API stdlib-only (`common/gpu_lock.py`, thống nhất với cleancode/tester/fixcode): `acquire() -> LockHandle`, `release(handle)` (chỉ khi token khớp),
   `read()`, `is_stale()` (chỉ báo), `verify_token(lock_path, token)` raise `GpuLockError`. Token vào container qua env `VLM_GPU_LOCK_TOKEN`, lock mount ro.
3. `docker run --name vlm-<run_id>` (KHÔNG `--rm`: runner `docker inspect` lấy exit_code + `.State.OOMKilled` → run.json, rồi `docker rm`; phân loại exit theo P009 §5) theo baseline security (user 1000, cap-drop ALL, no-new-privileges, 127.0.0.1 ports, env-file, image @sha256, `PYTHONPATH=/workspace/src`).
4. Tail stdout/stderr → `logs/`, adapter parse → `metrics.jsonl` theo contract frontend; watchdog memguard; SIGTERM → container stop có grace để save ckpt.
5. Exit mapping: 0 → `completed`, 75 (memguard) → `interrupted`, khác → `failed`. Luôn release lock trong `finally`.

### 2.3 RunManifest (`common/runinfo.py`) — mở rộng run.json của P006
`stage, parent_run_id, base_model, base_model_revision, dataset_id, container_image, container_digest, git_commit, git_dirty,
config_hash, seed, deterministic, versions{torch,transformers,nemo_automodel,modelopt,vllm}, status, started_at, ended_at, peak_mem_gb`.

### 2.4 Mapping stage → tool
| Stage | Tool | Ghi chú |
|---|---|---|
| train | `automodel <recipe.yaml>` (AutoModel 26.08) | config pydantic render ra recipe YAML upstream + overrides |
| quantize | ModelOpt `hf_ptq.py` | FP8 / NVFP4 / AWQ theo support matrix model |
| eval/bench | vLLM OpenAI endpoint (7871) + harness của P002 | so sánh BF16/FP8/NVFP4 cùng 1 harness |
| serve | vLLM (mặc định) / TRT-LLM (7872) | `gpu_memory_utilization ≤ 0.3` |

## 3. Số đo spike S-3 (2026-10-12 16:30, PASS)
- Image `vlm-automodel:26.08-1` (FROM 26.08 arm64 digest + chmod o+rX), driver 580.142 forward-compat, non-root uid 1000, `--network none`, cap-drop ALL, HF offline (model + dataset local snapshot pinned).
- Recipe `configs/train/smoke/qwen3_5_0p8b_lora_medpix_spike.yaml` (LoRA dim 8/alpha 32, vision frozen, GBS 4 = LBS 1 × grad-acc 4, act. checkpointing, sdpa, triton LoRA) → 20 step, exit 0, OOMKilled=false, wall 310s.
- Thời gian: load + setup ~25s, **step 0 mất ~2m50s** (JIT Triton kernel LoRA/GatedDeltaNet lần đầu — không cache vì TRITON_CACHE_DIR=/tmp trong container → nên mount cache Triton dự án), sau đó ~2.3–3.5 s/step, **tps 390–690** (median ~550).
- CUDA mem (AutoModel `mem` = max allocated): **8.14 GiB step 0**, steady 3.1–4.9 GiB. Host RAM: xem số của bao-tester (outputs/perf/s3-20261012/).
- Loss 2.36 → 1.42 (20 step, nhiễu), grad_norm 9–18.
- Checkpoint: `checkpoints/epoch_0_step_{9,19}/` {model/adapter_model.safetensors + adapter_config.json + tokenizer/processor, optim/ (DCP), dataloader/, rng/, step_scheduler.pt, losses.json, config.yaml}, 61MB mỗi cái; `LATEST -> epoch_0_step_19` (symlink, cập nhật atomic); `max_recent_checkpoints=2` hoạt động. Với PEFT, `save_consolidated: final` chỉ ra adapter (không merge full model) → merge là node riêng.
- **Marker hoàn tất**: AutoModel ghi `<ckpt>/.incomplete` khi bắt đầu save, xoá sau khi publish (`lifecycle._publish_checkpoint`: cập nhật LATEST rồi `clear_checkpoint_incomplete`). Resume bỏ qua thư mục còn `.incomplete`.
- **Metrics local có sẵn**: `checkpoints/training.jsonl` (+ validation.jsonl), mỗi dòng {step, epoch, timestamp, loss, grad_norm, lr, mem, tps, tps_per_gpu, num_tokens_per_step, num_label_tokens} → adapter sang metrics.jsonl (contract frontend) chỉ cần đổi tên key, không cần parse stderr.
- Câu hỏi mở #2 (save_consolidated final): **đã trả lời — có**. #1 (hook should_stop mỗi step): chưa — cần đọc recipe/callback trong S-2.

## 4. Không chọn
- Cài NeMo/ModelOpt lên host venv: wheel aarch64 + CUDA 13 thiếu (TE, bitsandbytes, flash-attn) → mong manh.
- Re-implement training loop: mất lợi thế recipe đã CI trên GB10.
- Orchestrator nặng (Ray/Airflow/Kubeflow) cho 1 GPU: overkill.

## 5. Câu hỏi mở
1. AutoModel có hook mỗi step để poll `should_stop` (P007 #1)? → kiểm trong spike.
2. `checkpoint.save_consolidated` có nhận `final` trong 26.08? (P005) → kiểm trong spike.
3. Megatron-Bridge có cần trong scope v1 không (chỉ khi base model là Nemotron VL lớn)?

## Ảnh hưởng của RESET (2026-10-12, R-4)
- Hướng giữ nguyên: P008 trở thành lớp **job runner** dưới graph executor của P009 (mỗi node nặng = 1 job container qua runner này).
- `vlm <stage>` CLI giữ cho dev/test/CI; node của app và CLI dùng chung adapter (params → recipe chính chủ).
- Thêm: `Worker` protocol (thay được bằng FakeWorker), error code catalog cho thông báo tiếng Việt, cache_key/artifact hash (P009 §5).
- Spike GPU (§3) HOÃN theo PM; image `nemo-automodel:26.08` chỉ pull + ghi digest, không chạy.

## Phụ lục A — Chiến lược image AutoModel khi driver 580 không chạy được 26.08 (nháp, chờ research xác minh 26.02)
| | (a) Giữ driver 580: build `vlm-automodel` FROM 26.02 (CUDA 13.0) + `nemo_automodel` pin commit + transformers mới | (b) User nâng driver (≥ bản 26.08 cần, research đọc ra ~610) → dùng thẳng 26.08 |
|---|---|---|
| Công sức ban đầu | ~2–4 ngày: Dockerfile, giải xung đột deps, test VLM recipe + LoRA + checkpoint | ~0.5 ngày phía ta (re-test); user làm nâng cấp + reboot |
| Rủi ro kỹ thuật | Cài đè lên NGC: transformers mới vs nội bộ NeMo/TE, có thể kéo đè torch (fixcode cảnh báo); QLoRA bitsandbytes SM121 có thể vẫn thiếu; vượt ra ngoài ma trận NVIDIA đã test → lỗi lạ khó report upstream | Driver mới tương thích ngược với image CUDA cũ (NVIDIA driver backward compatible) → service khác thường không vỡ, nhưng cần downtime + reboot máy chung |
| Bảo trì dài hạn | Mỗi lần nâng AutoModel/transformers phải lặp lại vá + test; tự chịu trách nhiệm combo không chính chủ | Theo nhịp NGC chính chủ (recipe *_spark, CI trên GB10); lần sau lại có thể cần nâng driver theo container mới |
| Ảnh hưởng người khác | Không | Downtime toàn bộ GPU service (NIM, voice-agent vLLM, comfy-ui…) trong cửa sổ bảo trì |
| Gộp được | — | Có thể gộp cùng cửa sổ vá CVE-2026-24260 (nvidia-container-toolkit) mà security đang chờ user |
Nhận định backend (sơ bộ): nếu 26.02 thiếu recipe VLM cho base model đã chọn hoặc cần QLoRA → nghiêng **(b)**, gộp chung 1 cửa sổ bảo trì với bản vá toolkit; (a) chỉ hợp lý nếu 26.02 đã đủ cho base model được chọn và chỉ cần LoRA BF16.

### A.1 Xác minh của bao-research (2026-10-12; đọc registry nvcr.io + git tag Automodel, không pull, không chạy GPU)
| Câu hỏi | Kết quả |
|---|---|
| 26.02 chứa gì? | Build 2026-02-27 = **Automodel v0.3.0** (tag `9e9472f`, cùng ngày). Torch 2.10.0a0, CUDA 13.0.2, TE release_v2.11, venv `/opt/venv` (uv, `--system-site-packages`), arm64 digest `sha256:6ee69689…3f45` |
| Recipe VLM trong v0.3.0 | Có `FinetuneRecipeForVLM`; recipe: gemma3/3n, internvl3.5-4b, kimi-vl, ministral3, nemotron_parse, phi4-mm, qwen2.5-vl-3b, **qwen3-vl 4b/8b**, qwen3-vl-moe 30b/235b. **KHÔNG có Qwen3.5/3.6/3.8, Gemma 4, Nemotron Omni** |
| Transformers | v0.3.0: `transformers>=5.0.0` (bản thực tế trong image ❓). v0.4.0 pin `==5.5.0`, v0.5.0 `==5.8.1`, v0.6.0 `==5.12.1` |
| Vá được lên 26.02 không? | **Có, với v0.4.0**: v0.4.0 (28/04) pin `torch>=2.6,<=2.10.0`, khớp đúng torch 2.10.0a0 của 26.02, nên không kéo đè torch/TE. v0.4.0 thêm recipe `qwen3_5_{4b,9b}`, `qwen3_5_moe_35b`, `gemma4_{2b,4b,31b,26b_a4b}(+peft)`, `mistral4`. v0.5.0+ bỏ cận trên torch (runtime) nhưng code có thể dùng API torch mới, nên rủi ro hơn. Cài: `uv pip install --no-deps nemo_automodel==0.4.0` + `transformers==5.5.0` + kiểm `pip check`; cấm upgrade torch bằng constraint `torch==<bản trong image>` |
| Qwen3.6-27B / Qwen3.8-27B trên v0.4.0? | Qwen3.6 dùng cùng kiến trúc Qwen3.5 → khả năng chạy được với YAML tự viết ❓. Qwen3.8 và Nemotron Omni cần v0.5+/v0.6 ❓. Với dev/main model (Qwen3.5-2B/4B/9B) thì **v0.4.0 là đủ** |
| QLoRA | v0.4.0 có `components/quantization/qlora.py` (bitsandbytes) + recipe LLM QLoRA (có `*_qlora_spark.yaml` cho Llama 70B), nhưng kernel bnb trên sm_121 aarch64 ❓. QLoRA chỉ cần cho model ≥27B. 2B/9B dùng LoRA BF16 |
| Driver tối thiểu cho 26.08 | "610.43" chỉ là driver lúc build image (`CUDA_DRIVER_VERSION`), **không phải mức tối thiểu**. Image CUDA 13.x chạy trên nhánh 580 nhờ forward-compat của NGC. Báo cáo cộng đồng (Classmethod) chạy được 26.08 trên **580.159.03**; forum #379168 báo không chạy được vllm 26.04 trên 580.173 → ❓ chưa nhất quán, cần probe |
| Kênh cập nhật chính chủ | Trang DGX Spark release notes (bản hiện tại) ghi **DGX OS 7.5.0 + driver 580.159.03 + CUDA 13.0.2**. **Chưa có driver 595/610 chính chủ cho Spark** (forum: "595 not yet supported on DGX Spark"). Máy ta đang ở **580.142, cũ hơn bản chính chủ** |

### A.2 Khuyến nghị research (đồng ký với backend nếu đồng ý)
- **Phương án (b) như đang viết (nâng lên ~610) là không khả thi**: chưa có driver chính chủ cho Spark, tự cài driver ngoài DGX OS thì mất hỗ trợ.
- Sửa thành **(b') cập nhật DGX OS lên bản chính chủ hiện hành (driver 580.159.03, cùng nhánh 580)** qua kênh apt của DGX OS. Gộp chung cửa sổ bảo trì với bản vá nvidia-container-toolkit. Rủi ro thấp (cùng nhánh). Đây là cấu hình có báo cáo chạy được 26.08, nhưng vẫn phải probe sau khi cập nhật.
- **Ngay bây giờ: (a) = 26.02 + nemo_automodel v0.4.0 + transformers 5.5.0**. Đủ cho Qwen3.5-2B/4B/9B (dev + main) bằng LoRA BF16, không đụng hệ thống. (b') là quyết định của user, khi cần Qwen3.8/Nemotron Omni/QLoRA hoặc khi cập nhật bảo mật.
- Nguồn: https://docs.nvidia.com/dgx/dgx-spark/release-notes.html · https://forums.developer.nvidia.com/t/595-58-03-certified-linux-aarch64-arm64-display-driver-and-cuda-13-2-when-for-dgx-spark-gb10/364688 · https://forums.developer.nvidia.com/t/new-dgx-os-7-4-0/359550 · git tags v0.3.0–v0.6.0 của https://github.com/NVIDIA-NeMo/Automodel

### A.3 bao-backend đồng ký A.2 (2026-10-12)
- Đồng ý: **ngay bây giờ (a)** = image `vlm-automodel` build FROM `nemo-automodel:26.02` + `uv pip install --no-deps nemo_automodel==0.4.0 transformers==5.5.0` + constraint `torch==<bản trong image>` + `pip check` + smoke import trong `check.sh --container`. **(b')** = cập nhật DGX OS chính chủ (580.159.03), gộp cửa sổ vá toolkit — user quyết.
- Bổ sung 1: rẻ nhất để gỡ ❓ "26.08 có chạy trên nhánh 580 không": **probe vài giây** trên image 26.08 đã có sẵn (banner forward-compat + `torch.cuda.is_available()` + matmul/SDPA bf16, < 2GB RAM, có GPU_LOCK, không train). Nếu chạy được ngay trên 580.142 → bỏ được (a) lẫn (b'). Cần PM mở cho phép probe (RESET đang cấm chạy GPU).
- Bổ sung 2: (a) cần pull `nemo-automodel:26.02` (~20–25GB) → cần PM duyệt (vượt ngưỡng 20GB). Làm probe trước rồi mới pull.
- Bổ sung 3: chọn Qwen3.5 còn vướng ModelOpt (Qwen3.5 chỉ FP8 validated, không NVFP4/AWQ) — độc lập với chuyện image, đưa vào bảng chọn model của P001.

### A.4 Kết quả probe (bao-backend, 2026-10-12 16:15, PM duyệt; GPU_LOCK + kiểm RAM, không train)
Image `nemo-automodel@sha256:b0bd0e50…` trên driver **580.142** (không cập nhật gì):
- Banner: `CUDA Forward Compatibility mode ENABLED. Using CUDA 13.3 driver version 610.43.02 with kernel driver version 580.142` → **chạy được**.
- `torch 2.13.0a0 (nv26.06)`, CUDA 13.3, `cuda_available=True`, GB10 capability 12.1, arch_list sm_80…sm_120 + compute_120 (sm_121 chạy cubin sm_120).
- bf16 matmul 4096² sau warmup: **~40 TFLOPS**, kết quả finite; SDPA bf16 causal OK; peak alloc 0.2 GB. (Lần đo không warmup chỉ 6 TFLOPS — luôn warmup khi benchmark.)
- transformers 5.12.1, TE 2.14.1. Banner ghi "NVIDIA Release 26.06" dù tag 26.08 (khớp báo cáo Classmethod).
- ⚠️ **Chạy non-root (uid 1000, baseline security) KHÔNG import được `nemo_automodel`**: `/opt/Automodel/nemo_automodel` là `drwxrwx--- root:root`. `automodel` CLI ở `/opt/venv/bin`.
  Đề xuất: image dẫn xuất `vlm-automodel:26.08-<n>` = `FROM nemo-automodel@sha256:b0bd… ` + `RUN chmod -R o+rX /opt/Automodel` (+ LABEL project=vlm-training), giữ non-root. Phương án khác: chạy root + cap-drop ALL + no-new-privileges rồi chown output — kém hơn. Chờ security.
- Hệ quả: **không cần (a) pull 26.02 lẫn (b') cập nhật DGX OS cho mức "chạy được"**. Chưa chứng minh train end-to-end (recipe VLM, LoRA, checkpoint) — cần spike khi PM mở lại. (b') vẫn nên làm theo lịch vá CVE toolkit vì driver 580.142 cũ hơn bản chính chủ 580.159.03.
