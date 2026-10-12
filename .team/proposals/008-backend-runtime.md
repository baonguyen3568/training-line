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
3. `docker run --rm --name vlm-<run_id>` theo baseline security (user 1000, cap-drop ALL, no-new-privileges, 127.0.0.1 ports, env-file, image @sha256, `PYTHONPATH=/workspace/src`).
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

## 3. Số đo spike (P1-9) — TODO
(image digest, recipe, peak mem, s/step, tok/s, startup peak, các lỗi gặp)

## 4. Không chọn
- Cài NeMo/ModelOpt lên host venv: wheel aarch64 + CUDA 13 thiếu (TE, bitsandbytes, flash-attn) → mong manh.
- Re-implement training loop: mất lợi thế recipe đã CI trên GB10.
- Orchestrator nặng (Ray/Airflow/Kubeflow) cho 1 GPU: overkill.

## 5. Câu hỏi mở
1. AutoModel có hook mỗi step để poll `should_stop` (P007 #1)? → kiểm trong spike.
2. `checkpoint.save_consolidated` có nhận `final` trong 26.08? (P005) → kiểm trong spike.
3. Megatron-Bridge có cần trong scope v1 không (chỉ khi base model là Nemotron VL lớn)?
