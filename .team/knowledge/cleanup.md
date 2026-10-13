# Knowledge — Cleanup / Storage Ops (bao-cleanup)

Cập nhật: 2026-10-12. Phạm vi: VLM training trên DGX Spark GB10 (aarch64, 128GB unified memory), máy dùng chung.

## 0. Hiện trạng máy (đo 2026-10-12 16:xx)
| Mục | Giá trị | Ghi chú |
|-----|---------|---------|
| Disk `/` (nvme0n1p2) | 3.7T, dùng 1.6T, còn **2.0T** (44%) | home cùng partition |
| RAM | 121G total, used 78G, **available 43G**, swap 15G đang dùng 8G | rất chật cho model ≥27B BF16 |
| `~/.cache/huggingface` | **190G** — KHÔNG thuộc dự án (qwen3.8-flash-next-nvfp4 132G, Nemotron 21G, SD3.5…) | không được đụng |
| `~/.cache/pip` / `uv` | 6.9G / 6.1G | dùng chung user |
| Docker | 197 images 634G, build cache 59G, 19 containers | **0 image `vlm-*`** — toàn bộ là của người khác |
| CLI `hf` trên host | không cài | dùng `uvx --from huggingface_hub hf ...` hoặc trong venv dự án |

→ Kết luận: hiện tại dự án chưa chiếm gì. Rủi ro lớn nhất không phải disk mà là **RAM** (unified) và việc vô tình dọn nhầm tài nguyên người khác.

## 1. Nguyên tắc vàng (áp dụng cho dự án)
1. **Scope cứng**: chỉ đường dẫn dưới `/home/ntcai/Trainning-model` + docker object có prefix `vlm-` **hoặc** label `project=vlm-training`. Mọi script phải tự kiểm tra `realpath` nằm trong root trước khi xoá (chống symlink thoát ra ngoài).
2. **Dry-run mặc định**, xoá thật cần `--apply`; in danh sách + tổng dung lượng trước.
3. **Không bao giờ** `docker system prune`, `docker image prune -a` không filter, `docker builder prune` toàn cục (build cache 59G là của các project khác, builder `buildx_buildkit_comfyui-publisher0` cũng không phải của ta).
4. Xoá > 20GB hoặc bất kỳ thứ gì "best/final/released" → báo PM + bao-security trước.
5. Không xoá thứ đang được dùng: kiểm tra GPU lock bằng `uv run python -m vlm_pipeline.common.gpu_lock status` (file ở `.team/lock/GPU_LOCK`), `lsof +D`/`fuser`, tiến trình train còn sống.

## 2. Hugging Face cache
Nguồn: https://huggingface.co/docs/huggingface_hub/guides/manage-cache (huggingface_hub v2.x)
- Layout: `hub/<type>--<org>--<name>/{blobs,snapshots,refs,trees,.no_exist}`; snapshots là symlink → blobs. Từ v2.x có **shared xet blobs** `hub/blobs/<prefix>/<xet_hash>` dedupe giữa các repo.
- **Pitfall**: `rm -rf models--X` thủ công hoặc tool cũ có thể xoá blob shared còn được repo khác dùng → chỉ dùng `hf cache rm` / `hf cache prune` bản mới.
- Lệnh chuẩn:
  - `hf cache ls [--revisions] --filter "size>1GB" --filter "accessed>30d" --sort size:desc --format json`
  - `hf cache rm <model/org/name|rev_hash|hf://models/.../file> --dry-run` rồi `-y`
  - `hf cache prune --dry-run` — xoá revision không còn ref, file `.incomplete`, shared blob mồ côi.
  - `hf cache verify <repo>` — checksum khi nghi cache hỏng.
  - Luôn kèm `--cache-dir` khi làm việc với cache dự án.
- `xet/` (shard-cache, staging) chỉ phục vụ upload, soft limit 16GB (`HF_XET_SHARD_CACHE_SIZE_LIMIT`), có thể xoá toàn bộ an toàn.
- `datasets` Arrow cache (`HF_DATASETS_CACHE`) KHÔNG được `hf cache ls` thống kê → đo bằng `du` riêng; dọn bằng `Dataset.cleanup_cache_files()` hoặc xoá thư mục.
- Env: `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_XET_CACHE`, `HF_HUB_OFFLINE=1` (khi eval/serve để reproducible).
- **Quyết định đề xuất**: dự án dùng `HF_HOME=/home/ntcai/Trainning-model/.cache/hf` (gitignored) → cô lập, dọn được an toàn, đo được chính xác. Đánh đổi: không tái dùng 190G cache global — nếu cần model đã có sẵn ở global (vd Nemotron 3.5 Lightning) thì **copy có chủ đích** hoặc đọc read-only qua `--cache-dir`/`local_files_only`, không bao giờ xoá trong global.

## 3. Checkpoints
Nguồn: https://docs.nvidia.com/nemo/automodel/development/checkpointing ; NeMo-RL `keep_top_k` (https://docs.nvidia.com/nemo/rl/0.3.0/_modules/nemo_rl/utils/checkpoint.html)
- **NeMo AutoModel**: `checkpoints/epoch_<E>_step_<S>/{model,optim,dataloader,rng,config.yaml}` + symlink `LATEST`, `LOWEST_VAL`.
  - `checkpoint.max_recent_checkpoints: N` → giữ cửa sổ N bản gần nhất; target của `LATEST`/`LOWEST_VAL` luôn được giữ. `null` = giữ tất cả (mặc định — nguy hiểm!). Chỉ chạy được trên local FS.
  - `checkpoint.save_consolidated: final` (khuyên dùng) — chỉ export HF safetensors ở cuối; `every` nhân đôi dung lượng mỗi lần save.
  - `optim/*.distcp` chiếm ~2× weights (Adam fp32 m,v) → checkpoint full = ~3× model BF16. Đây là thứ đầu tiên nên xoá ở checkpoint không còn cần resume.
  - PEFT/LoRA: adapter chỉ vài MB–vài trăm MB trong `model/`; cần base model **giữ nguyên revision** để reload → pin revision base model, không prune revision đó khỏi HF cache.
- **NeMo-RL**: `checkpointing.keep_top_k`, `metric_name`, `higher_is_better`.
- **NeMo 2.0/Lightning (Megatron-Bridge)**: `ModelCheckpoint(save_top_k, save_last, monitor)`.
- **Retention đề xuất** (chi tiết ở Proposal 005): giữ `LATEST` + `best` (LOWEST_VAL) + N=2 gần nhất trong lúc train; khi run kết thúc: giữ consolidated final + best, **strip `optim/` khỏi mọi bản không phải LATEST**, xoá intermediate sau 7 ngày nếu run không được đánh dấu `keep`.
- Marker file `KEEP` trong thư mục run → script dọn không bao giờ đụng.

## 4. Docker (chỉ objects của dự án)
Nguồn: https://docs.docker.com/reference/cli/docker/system/prune/ (filter `label`, `label!=`, `until`)
- Mọi image/container/volume dự án: tên `vlm-*` **và** `LABEL project=vlm-training` (Dockerfile) / `--label project=vlm-training` (run/compose). Label cho phép filter an toàn; prefix tên cho con người đọc.
- Lệnh an toàn:
  - `docker ps -a --filter name=^vlm- --filter status=exited`
  - `docker container prune --filter label=project=vlm-training --filter until=72h`
  - `docker image prune --filter label=project=vlm-training --filter until=168h` (chỉ dangling); image có tag thì liệt kê `docker images --filter label=project=vlm-training` rồi `docker rmi` từng cái.
  - Build cache: dùng builder riêng `docker buildx create --name vlm-builder` → `docker buildx prune --builder vlm-builder --filter until=168h` chỉ ảnh hưởng builder của ta. **Không** prune default builder.
  - Volume: `docker volume ls --filter label=project=vlm-training`.
- Image NGC base (nvcr.io/nvidia/pytorch, nemo…) 20–30GB: giữ đúng 1 tag đang được pin trong ADR; tag cũ xoá khi ADR đổi — nhưng **kiểm tra image không bị container khác (ngoài dự án) dùng** (`docker ps -a --filter ancestor=<img>`).
- Container NGC nên chạy `--rm` cho job một lần; log lấy ra file trong `outputs/logs/`.

## 5. Unified memory (GB10) — phần RAM
Nguồn: https://build.nvidia.com/spark/vlm-finetuning/troubleshooting ; https://build.nvidia.com/spark/unsloth/troubleshooting ; https://github.com/natolambert/dgx-spark-setup
- CPU+GPU chung 128GB; `nvidia-smi` báo `memory.used [N/A]` → **đo bằng `free -g` / `/proc/meminfo` (MemAvailable)**.
- Page cache (đọc safetensors/dataset lớn) chiếm RAM; playbook NVIDIA khuyên `sync; echo 3 > /proc/sys/vm/drop_caches` — **cần sudo, ảnh hưởng toàn host** → theo CLAUDE.md phải xin PM/user, không tự chạy.
- OOM trên Spark có thể treo cả máy (không chỉ kill job) → ảnh hưởng service người khác. Ngưỡng cảnh báo đề xuất: MemAvailable < 20GB → cảnh báo PM; < 10GB → khuyến nghị dừng job dự án (không bao giờ đụng process người khác).
- Swap đang dùng 8/15GB sẵn → hệ đã có áp lực bộ nhớ.
- Ước lượng trước khi train: weights + grads + optimizer + activations; model 27-35B BF16 full-FT là **không khả thi** với ~43GB available → LoRA/QLoRA hoặc model 2-8B.

## 6. Các nguồn tạo rác khác
- `outputs/runs/<run_id>/` (layout chốt ở P005 D2): logs, tensorboard/wandb offline (`wandb/` — `wandb sync --clean`), eval predictions, quantized exports (TRT-LLM engines rất lớn, rebuild được → xoá được).
- TensorRT-LLM engine / timing cache, Triton/torch compile cache (`~/.cache/torch`, `TORCHINDUCTOR_CACHE_DIR`) → đặt vào `.cache/` của dự án.
- `data/raw` (tải lại được) vs `data/processed` (tốn công tạo) — retention khác nhau; WebDataset/Energon shards giữ, intermediate curation xoá.
- `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.venv` — đã/ sẽ gitignore.
- Core dump / `*.nsys-rep` / `*.ncu-rep` profiler output có thể vài GB.

## 7. Công cụ
- `du -xsh`, `ncdu -x` (nếu có), `find -size +1G -printf`, `fuser`/`lsof`, `df -h`, `free -g`.
- `hf cache ls|rm|prune|verify` (qua `uvx --from huggingface_hub hf`).
- **Kho blob chung của dự án `.cache/hf/hub/blobs/<xx>/<name>`** (quan sát 2026-10-13): do chính huggingface_hub tạo khi tải
  (marker `blobs/.huggingface-shared-blobs`, kèm `<name>.refs` liệt kê repo blob trỏ vào, và `<name>.lock`; tải qua hf_xet,
  có `.cache/hf/xet/`). `models--*/blobs/<etag>` là symlink vào kho này. `<name>` KHÔNG phải sha256 nội dung (đoán là Xet
  hash, chưa kiểm) → chỉ là khoá dedupe; toàn vẹn kiểm bằng sha256 = etag LFS. **Không xoá** file trong `blobs/<xx>/` chỉ vì
  không thấy symlink: phải xét `.refs`, và chỉ dọn bằng `hf cache rm/prune` của đúng phiên bản huggingface_hub đã tạo kho.
- `docker system df -v` (chỉ đọc — OK), docker filters.
- Script dự án (sắp viết): `scripts/cleanup/disk_report.sh`, `scripts/cleanup/retention.py --dry-run` (mặc định) / `--apply`.

## 8. Pitfalls tổng hợp
- Xoá checkpoint đang được ghi (async save `checkpoint.is_async`) → hỏng. Kiểm tra file `.tmp`/process trước.
- Xoá target của symlink `LATEST`/`LOWEST_VAL` → resume vỡ. Luôn resolve symlink trước.
- `rm -rf $VAR/` với biến rỗng → xoá nhầm root. Script dùng `set -euo pipefail` + `${VAR:?}` + kiểm tra prefix.
- `docker image prune -a` xoá image không có container chạy — sẽ xoá image của người khác đang Exited.
- `hf cache` cũ (huggingface-cli delete-cache) + shared xet blobs → xoá blob đang dùng.
- `du` của HF cache đếm symlink sai nếu dùng `-L`.
- drop_caches cần root → không tự làm.

## 9. Checklist trước khi xoá
- [ ] Đường dẫn resolve nằm trong project root / object có label `project=vlm-training`
- [ ] GPU lock (`gpu_lock status`) không giữ run liên quan / process đang mở file (`fuser -v`)
- [ ] Không có marker `KEEP`, không là target của `LATEST`/`LOWEST_VAL`/best
- [ ] Đã chạy dry-run, có tổng dung lượng
- [ ] > 20GB hoặc artifact "release" → PM + security đồng ý
- [ ] Ghi log vào `.team/log/<ngày>.md` (đã xoá gì, bao nhiêu GB)

## 10. RESET — app no-code node-graph (2026-10-12)
- ComfyUI cache mode: classic (mặc định, dọn mạnh), `--cache-lru N` (giữ N output node), `--cache-none`, RAM-pressure (`--cache-ram`, ngân sách mặc định ~16GB); in-memory, chỉ dọn sau khi prompt chạy xong. Nguồn: mirror execution.py (https://huggingface.co/vidfom/LTX-2-S/blob/main/ComfyUI/execution.py), https://mintlify.wiki/Comfy-Org/ComfyUI/concepts/execution. Cờ phụ thuộc phiên bản, cần kiểm lại trên repo chính thức.
- Áp vào dự án: output node nặng (GB) → cache trên disk, content-addressed, LRU, có trần; RAM cache chỉ cho output nhỏ. Xem P005 §6.
