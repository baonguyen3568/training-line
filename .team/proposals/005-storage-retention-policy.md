# Proposal 005 — Storage & Retention Policy

- Tác giả: bao-cleanup · Trạng thái: **ACCEPTED → ADR-005** (điều kiện: verify save_consolidated; re-check quota khi chốt model list) (backend, security, research, tester, frontend, cleancode) → PM chốt ADR
- Ngày: 2026-10-12 · Knowledge nền: `.team/knowledge/cleanup.md`

## 1. Bối cảnh & ràng buộc (đo thực tế 2026-10-12)
- Disk `/` còn ~2.0T/3.7T, **dùng chung** (docker của người khác 634G images + 59G build cache; HF cache global 190G không phải của ta).
- RAM unified 121G, MemAvailable ~43G, swap đã dùng 8/15G. OOM trên GB10 có thể treo cả host.
- Dự án hiện chiếm ~0 (chưa có image `vlm-*`, chưa có checkpoint).
- Mục tiêu: (a) không bao giờ ảnh hưởng tài nguyên người khác, (b) footprint dự án có trần rõ ràng, (c) dọn dẹp tự động nhưng an toàn (dry-run mặc định), (d) không mất artifact quan trọng.

## 2. Quyết định đề xuất

### D1. Mọi cache nằm trong project root (`.cache/`, gitignored)
```bash
# .env.example (nạp bởi CLI/compose)
HF_HOME=/home/ntcai/Trainning-model/.cache/hf          # hub/, datasets/, xet/
TORCHINDUCTOR_CACHE_DIR=/home/ntcai/Trainning-model/.cache/torchinductor
TRITON_CACHE_DIR=/home/ntcai/Trainning-model/.cache/triton
UV_CACHE_DIR=  # giữ mặc định ~/.cache/uv (dùng chung user, chỉ là wheel, không dọn)
```
- Container mount `.cache/hf` → `/root/.cache/huggingface` (hoặc set `HF_HOME` trong container).
- Global `~/.cache/huggingface`: **read-only reference**. Nếu cần model đã có ở đó (vd Qwen3.8-27B-NVFP4 cho baseline eval), load bằng `cache_dir=` + `local_files_only=True`; không ghi, không xoá.
- Lý do: đo chính xác footprint, dọn bằng `hf cache rm/prune --cache-dir` mà không có rủi ro đụng repo/blob người khác (shared xet blobs). Đánh đổi: có thể tải trùng vài model → chấp nhận.
- Eval/serve chạy với `HF_HUB_OFFLINE=1` + revision pin để reproducible.

### D2. Layout run directory (đã gộp với bao-backend + bao-frontend)
```
outputs/runs/<run_id>/                  # run_id = YYYYMMDD-HHMM-<recipe>-<model_short>-<hash4>
  run.json          # config resolve + git sha + versions + status (running|completed|failed|discarded) + parent_run_id
  metrics.jsonl  system.jsonl  events.jsonl   # contract của frontend — giữ mãi
  KEEP              # (tuỳ chọn) marker: retention không đụng
  logs/             # stdout, tensorboard (giữ mãi, nén sau 30 ngày)
  checkpoints/      # epoch_<E>_step_<S>/, LATEST, LOWEST_VAL
  export/           # consolidated HF safetensors / LoRA adapter / ModelOpt quantized (FP8, NVFP4)
  engines/          # TRT-LLM engines (regenerable)
  eval/             # predictions + metrics json
outputs/debug/BUG-NNN/  # fixcode: giữ tới khi bug VERIFIED + 14 ngày
outputs/smoke/      # smoke test của tester — xoá ngay sau test (tester tự dọn; retention quét >3 ngày)
outputs/system/    # disk_report.json, cleanup.jsonl
data/raw/           # regenerable
data/processed/     # đã curate/format (giữ; versioned theo tên thư mục)
benchmarks/         # kết quả benchmark tổng hợp (commit git, giữ vĩnh viễn)
```
- Container chạy `--user $(id -u):$(id -g)` để file không root-owned (nếu image không cho phép → backend chown trong project sau job). Script retention gặp file không xoá được → báo lỗi, KHÔNG sudo.

### D3. Retention checkpoint
**Trong lúc train** (cấu hình recipe mặc định):
| Framework | Cấu hình |
|-----------|----------|
| NeMo AutoModel | `checkpoint.max_recent_checkpoints: 2`, `checkpoint.save_consolidated: final`* , `checkpoint.model_save_format: safetensors` |
| NeMo-RL | `checkpointing.keep_top_k: 2` + `metric_name`, `higher_is_better` |
| Megatron-Bridge / Lightning | `save_top_k: 2`, `save_last: true`, `monitor: val_loss` |

\* Docs AutoModel hiện tại (docs.nvidia.com/nemo/automodel/development/checkpointing) ghi `final | false | every`, trong đó `true` = legacy alias của `every` (export mỗi lần save → tốn gấp đôi). Recipe upstream (qwen3_5_4b.yaml) dùng `true` → **bao-backend xác minh trong container 26.08**; nếu bản 26.08 chỉ có bool thì dùng `true` nhưng tính dung lượng như `every`.

**Thứ tự ưu tiên khi một checkpoint trùng nhiều vai trò**: `KEEP` > `LATEST` > `LOWEST_VAL`/best > khác. Ví dụ LATEST == LOWEST_VAL → áp luật LATEST (giữ optim 7 ngày). Script resolve symlink trước rồi gán vai trò cao nhất.
LoRA: adapter có cả trong `checkpoints/epoch_*/model/` và `export/` → tính dung lượng theo inode (không đếm trùng), không coi bản trong checkpoint là "thừa" chỉ vì đã có ở export.

**Sau khi run kết thúc** (`scripts/cleanup/retention.py`, dry-run mặc định). Chỉ áp dụng khi `run.json.status == completed` (hoặc `discarded`); run `running`/`failed`/`interrupted` chỉ được báo cáo, không strip/xoá — cần để resume:
| Loại | Giữ | Xoá |
|------|-----|-----|
| Target của `LATEST` | đủ (để resume) trong 7 ngày, sau đó strip `optim/`,`dataloader/`,`rng/` | — |
| Target của `LOWEST_VAL` / best | weights + config | `optim/` ngay khi run xong |
| Checkpoint khác | — | sau 7 ngày (nếu không `KEEP`) |
| `export/` (consolidated, adapter, quantized) | **giữ** đến khi run bị đánh dấu discard | — |
| LoRA adapter | luôn giữ (vài trăm MB) | — |
| `eval/` raw predictions | run gắn với báo cáo/ADR (hoặc `KEEP`) | còn lại sau 14 ngày (metrics json luôn giữ) |
| `engines/` | 14 ngày kể từ lần dùng cuối | rebuild được |
| run thất bại/smoke (`*-smoke-*`, không có eval) | — | sau 3 ngày |

### D4. HF cache dự án
- Hàng tuần (hoặc khi project footprint > quota): `hf cache prune --cache-dir .cache/hf/hub --dry-run` → apply.
- Model không truy cập > 30 ngày và không được pin trong config đang dùng → đề xuất xoá (báo team, không tự xoá nếu > 20GB).
- Dataset eval trong regression gate (MMMU val, MMBench dev, DocVQA val, ChartQA test, OCRBench, AI2D, MathVista testmini — chốt ở P002) **pin commit sha, không prune**. Danh sách pin: `configs/cleanup/pinned.yaml`.
- **Không xoá revision base model mà một LoRA adapter còn giữ phụ thuộc** (retention đọc `run.json (`base_model`, `base_model_revision`)` để biết pin).
- `.cache/hf/xet` xoá được bất kỳ lúc nào.

### D5. Docker
- Mọi Dockerfile: `LABEL project=vlm-training`; image tag `vlm-<component>:<version>`; container `--name vlm-<job>`; job một lần `--rm`; compose project name `vlm`.
- Build bằng builder riêng: `docker buildx create --name vlm-builder --driver docker-container` → prune chỉ builder này.
- Dọn (dry-run → apply): container exited `vlm-*` > 72h; dangling image label dự án > 7 ngày; image tag cũ không còn trong ADR (kiểm tra `docker ps -a --filter ancestor=` không có container ngoài dự án).
- Base NGC image (20-30GB): chỉ giữ tag đang pin trong ADR-001 + 1 tag trước (rollback).
- **Image dự án hiện có** (theo `docker/IMAGES.lock`, security APPROVED): `nvcr.io/nvidia/nemo-automodel:26.08` @ `sha256:b0bd0e50…` (arm64 `sha256:70b4e9df…`), 29.4 GB — **giữ**; tính vào footprint dự án; không có label → nhận diện qua IMAGES.lock (D7 mục 10).

### D6. Quota & cảnh báo
| Chỉ số | Warn → PM | Critical |
|--------|-----------|----------|
| Project footprint (`.cache`+`outputs`+`data`+images vlm) | > 500 GB | > 800 GB → dừng download mới, chạy retention |
| Disk free `/` | < 500 GB | < 200 GB |
| MemAvailable | < 20 GB | < 10 GB → khuyến nghị dừng job của dự án |
- `scripts/cleanup/disk_report.py` → in bảng + ghi `outputs/system/disk_report.json` (cho dashboard). Chạy trước mỗi job nặng (gắn vào bước tạo GPU_LOCK) và theo yêu cầu.
- Download > 20GB vẫn phải báo PM trước (CLAUDE.md).
- `outputs/system/disk_report.json` có `schema_version`, `ts`, `disk{total,used,free}`, `mem{total,available,buff_cache,swap_used}`, `project{cache_hf,outputs,data,docker_vlm}`, `level` (ok|warn|critical). Phân loại "thuộc dự án": path dưới project root, docker object tên `vlm-*` hoặc label `project=vlm-training`; RAM "của dự án" = RSS các process có cgroup/container `vlm-*` hoặc PID trong `.team/GPU_LOCK` (ước lượng, ghi rõ trên UI).

### D8. Pre-flight check trước khi lấy GPU_LOCK (PM yêu cầu)
**Một code path duy nhất** (thống nhất backend + fixcode/P007): logic ở `src/vlm_pipeline/common/memguard.py` — `preflight(est_peak_gb) -> Verdict` (+ `watchdog` của P007). `scripts/cleanup/preflight.py --est-peak-gb <N>` và job runner của backend chỉ gọi vào; in JSON verdict ra stdout (ghi vào events.jsonl). Exit code: `0` ok, `2` thiếu RAM/swap, `3` thiếu disk, `4` lock bận. Gọi trong bước acquire GPU_LOCK (skill `backend-gpu-job`). **Từ chối chạy (không tạo lock)** nếu:
| Điều kiện | Ngưỡng |
|-----------|--------|
| `MemAvailable` | < `est_peak_gb` + margin, margin = max(10 GB, 20% × est_peak_gb) (PM chốt chung P005/P007) |
| `MemAvailable` tuyệt đối | < 20 GB (sàn cứng cho mọi job nặng) |
| `SwapFree` | < 2 GB (host đã áp lực bộ nhớ) |
| Disk free `/` | < kích thước checkpoint dự kiến × (max_recent_checkpoints+1) + 50 GB |
| `.team/GPU_LOCK` | đã tồn tại và PID còn sống (exit 4). PID chết = stale lock → chỉ báo PM, **không bao giờ tự xoá lock** |
- `est_peak_gb` lấy từ memory budget của research (skill `research-model-selection`) và ghi vào `run.json`; job không khai báo → mặc định coi như cần 40 GB.
- Ví dụ hiện tại (MemAvailable 43G): LoRA 4B (~16 GB peak) → cần ≥ 26 GB → OK; full-FT 8B (~70 GB) → từ chối.
- Không có `drop_caches`/sudo (quyết định tạm của PM). Từ chối thì in số liệu MemAvailable/swap và hướng dẫn báo PM.
- `--force` chỉ dùng khi PM đồng ý; ghi lý do vào log.
- Runtime (watchdog P007): abort khi MemAvailable < 8 GB (fixcode có quyền phản biện con số nếu có dữ liệu).
- Job dừng bởi watchdog (exit 75, P007) → `run.json.status = interrupted` → retention coi như running/failed (không strip optim/, để resume).

### D7. An toàn khi xoá (ràng buộc cho mọi script dọn)
1. Dry-run mặc định; `--apply` mới xoá; in danh sách + tổng GB.
2. `realpath` phải nằm dưới project root; từ chối symlink trỏ ra ngoài.
3. Bỏ qua: `KEEP`, target `LATEST`/`LOWEST_VAL`, file đang mở (`fuser`), run đang giữ `.team/GPU_LOCK`, checkpoint có file `.tmp`/đang async save.
4. Tổng > 20GB hoặc có `export/` → cần xác nhận PM + bao-security (flag `--approved-by`). *Trong app: theo `storage.approval_policy` (§6 R2) — máy dev chung của dự án = `threshold` (giữ nguyên luật này); bản cài cá nhân = `none`.*
5. Mọi lần `--apply` append log `.team/log/YYYY-MM-DD.md` + `outputs/system/cleanup.jsonl`.
6. **Whitelist không bao giờ xoá** (yêu cầu bao-frontend): `run.json`, `metrics.jsonl`, `system.jsonl`, `events.jsonl`, `logs/`, toàn bộ `benchmarks/`.
7. Mỗi lần prune checkpoint/export → append vào `outputs/runs/<run_id>/events.jsonl`: `{"event":"checkpoint_pruned","path":...,"action":"strip_optim|delete","bytes":...,"ts":...,"by":...,"approved_by":...}` để dashboard hiển thị "đã xoá" thay vì link hỏng.
8. **Không bao giờ xoá (yêu cầu bao-security)**: `configs/security/**`, `docker/IMAGES.lock`, manifest sha256 của dataset shard/model snapshot, SBOM/scan report (`benchmarks/security/`, `outputs/runs/<run_id>/provenance/`). Run "release": giữ `provenance/` kể cả khi weights trung gian đã xoá.
9. Script không đọc/in `.env*`, `*.pem`, `~/.cache/huggingface/token`; loại chúng khỏi mọi danh sách in ra.
10. Image base NGC (không có label/prefix): chỉ coi là của dự án nếu digest có trong `docker/IMAGES.lock`. Xoá cần 2 bước: (a) digest có trong IMAGES.lock và không còn được ADR pin, (b) `docker ps -a --filter ancestor=<id>` rỗng (kể cả container ngoài dự án) → PM xác nhận. Image không có trong lock, không `vlm-`, không có label → không đụng (vd nemo-speech của người khác).
11. Cấm tuyệt đối trong code: `docker system prune`, `docker image prune -a` không label, prune default builder, `sudo`, đường dẫn ngoài root.

## 3. Deliverables (Phase 2, sau ADR)
- `scripts/cleanup/disk_report.py`, `scripts/cleanup/retention.py` (pure Python, stdlib + pydantic config `configs/cleanup/retention.yaml`), unit test với thư mục giả (tmp_path) — nhờ bao-tester review.
- `.env.example` cache vars; snippet LABEL cho Dockerfile (bao-backend áp dụng).
- `retention.py` dry-run xuất JSON (schema_version 1): `runs[].items[]` với `roles`, `action` (keep|strip_optim|delete), `reclaim_gb`, `rule`, `reason`, `blocked_by`; top-level `total_reclaim_gb`, `needs_approval` (theo `storage.approval_policy` của deployment — xem §6 R2). Dùng cho view retention của bao-design (P006 v1.1).
- Plan JSON v2 (để phục vụ panel no-tech, P010 §9): thêm `items[]` phẳng bên cạnh `runs[]`, mỗi item: `item_id`, `group` (downloaded_model|trained_model|dataset|temp_result), `display_name` (chỉ tên raw do user/model đặt, else null) | `display_key` + `display_params` (nhãn hệ thống sinh, i18n), `size_gb`, `last_used`, `used_by[]` (tên workflow), `action`, `regenerable`, `regen_estimate_min`, `redownload_gb`, `locked`, `lock_reason_code`, `consequence_code` + params (UI dịch sang lời), `confirm_required`, `confirm_text`, `event_type` (checkpoint_pruned|cache_evicted|hf_model_removed|dataset_removed).
- Kiểm tra định kỳ git index (yêu cầu security): file >5MB hoặc `.env*` bị stage/commit → báo security + PM (`git ls-files -s` + `git diff --cached --name-only`).
- Pre-commit `check-added-large-files --maxkb=1000` (chốt trong Proposal 004 §4).

## 4. Câu hỏi mở
1. ~~(backend) tần suất save~~ → đã trả lời: `ckpt_every_steps` 100–500, consolidated final là input cho ModelOpt; strip optim chỉ khi status=completed.
2. ~~(security) artifact audit~~ → đã trả lời (D7 mục 8–10).
3. (research) Danh sách model dự kiến + size → kiểm tra quota 500GB có hợp lý không.
4. (PM/user) Quota 500/800GB có chấp nhận? `drop_caches` trước job lớn cần sudo — user có muốn cho phép không, hay chỉ cảnh báo?
5. ~~(tester) dataset pin~~ → đã trả lời (xem D4); Video-MME ngoài regression, tải khi cần (báo PM nếu >20GB).

## 5. Phương án đã cân nhắc và loại
- **Dùng HF cache global**: tiết kiệm tải trùng nhưng không dọn được an toàn, footprint không đo được → loại.
- **Giữ tất cả checkpoint** (mặc định AutoModel): model 8B full-FT ~ 50GB/checkpoint với optim → vài chục bản = TB → loại.
- **Cron tự xoá không cần người**: rủi ro cao trên máy chung giai đoạn đầu → chỉ tự động phần an toàn (xet cache, `.incomplete`, smoke runs); phần còn lại dry-run + báo cáo.

## 6. Ảnh hưởng của RESET (2026-10-12) — app no-code node-graph
ADR-005 **giữ nguyên** (scope, dry-run, D7 an toàn, D8 preflight). Mọi quy tắc vẫn áp dụng, chỉ đổi *ai* kích hoạt: người dùng no-tech qua UI thay vì team qua CLI. Bổ sung:

### R1. Node output cache = loại dữ liệu mới (lớn nhất tiềm năng)
Khác ComfyUI: output node của ta là dataset, checkpoint, model quantized (GB–chục GB), không phải ảnh/latent trong RAM.
- ComfyUI có 4 chế độ cache (classic / LRU `--cache-lru N` / none / RAM-pressure), tất cả là **in-memory** và chỉ dọn sau khi prompt chạy xong. Trên GB10 (UMA, ~40G thực) giữ output nặng trong RAM = nguy cơ OOM treo host → **không áp dụng nguyên mẫu**.
- **Chốt với bao-backend (khớp P009 §5)**: cache trên disk, **không nhân đôi dữ liệu**. Output node nặng chỉ nằm ở `outputs/runs/<run_id>/` (mỗi node nặng 1 run dir). Index cache = bảng `artifacts` trong **SQLite `outputs/app.db` (WAL) — nguồn sự thật duy nhất** (cache_key, path, hash manifest, parent ids, `last_hit_at`, `pinned`, `state`, `evicted_at`); không có `entry.json`. `cache_key = H(type_id, version, params_canonical, input artifact hashes, image_digest)`.
- **Evict = xoá bytes, không xoá lịch sử**: chỉ xoá dữ liệu nặng (`checkpoints/`, `engines/`, output nặng), giữ whitelist D7 (`run.json`, `*.jsonl`, `logs/`) + `export/` do người dùng đặt tên; `run.json.status = evicted` + event `cache_evicted`.
- **Evict 3 bước** (tránh race với executor): transaction đặt `state=evicting` (executor coi là miss) → xoá file → `state=evicted`. Khi cache hit, executor kiểm path tồn tại + hash manifest; lệch → miss + cảnh báo.
- **Lineage**: không evict artifact có con còn sống với cạnh `requires_parent_at_load` (vd LoRA adapter → base là output full-FT của node trước). Bản quantized/merged tự chứa → cha evict được.
- Node tải HF tham chiếu `.cache/hf` (không bản thứ hai). `cacheable=false`: `serve.vllm`/Endpoint, `playground.chat`, `export.*`, nguồn local chưa có manifest hash.
- Trần 200 GB chỉ tính **run evict được** (không pinned / `KEEP` / export / có con `requires_parent_at_load`).
- RAM cache in-process chỉ cho output nhỏ (metadata, preview, metrics), giới hạn cứng (vd ≤ 2 GB tổng, LRU).
- Output "nặng" mà người dùng đã lưu/đặt tên (model đã train, bản quantized dùng để export) **không phải cache** → chuyển thành *artifact* trong `outputs/runs/<run_id>/export/`, theo retention D3 bình thường.
- Retention cache: LRU theo `last_used`, trần riêng (đề xuất 200 GB, nằm trong quota 500 GB); entry đang được graph mở/đang chạy hoặc `pinned` không bị xoá; xoá cache luôn an toàn về mặt dữ liệu (tính lại được) nhưng tốn thời gian → UI hiển thị "tính lại mất ~X phút".
- **Điều kiện PM (chốt 2026-10-12)**:
  1. 200 GB là **trần trên, không phải phần giữ chỗ**. Khi footprint dự án chạm warn (500 GB), cache node là thứ **bị evict LRU đầu tiên**, trước khi đụng tới checkpoint/export. Thứ tự giải phóng: cache node → `.cache/hf/xet` + `hf cache prune` (revision mồ côi) → engines (rebuild được) → eval raw > 14 ngày → mới tới checkpoint theo D3.
  2. Trần **cấu hình được trong Settings của app** (`storage.node_cache_max_gb`), vì máy của người dùng khác có ổ khác. Mặc định (PM chốt) = min(200 GB, 10% dung lượng trống lúc cài đặt). Validate: không cho đặt vượt quota dự án.
- Cần thống nhất với bao-backend + bao-frontend (Proposal 009: executor) về định nghĩa cache key và cờ `cacheable` per node type (vd node download HF → cache = HF cache, không nhân đôi).

### R2. Panel "Dung lượng" cho người dùng no-tech
- Nguồn dữ liệu = `disk_report.json` + `retention.py` dry-run JSON (đã chốt shape với design/frontend) + danh sách cache node.
- Hiển thị theo ngôn ngữ người dùng: "Model đã tải", "Model bạn đã train", "Dữ liệu", "Kết quả tạm (có thể tính lại)", kèm dung lượng + lần dùng cuối; KHÔNG hiện đường dẫn/khái niệm optim/DCP trừ chế độ nâng cao.
- **Nút xoá an toàn** (khác dashboard team, nơi chỉ copy lệnh dry-run):
  1. Bấm xoá → backend chạy dry-run → dialog liệt kê chính xác thứ sẽ bị xoá + dung lượng + hậu quả ("workflow 'LoRA demo' sẽ phải tải lại model 8 GB").
  2. Gõ tên để xác nhận **chỉ khi** mục không tính lại được (model đã train) hoặc phải tải lại > 20 GB; mục tính lại được (kết quả tạm) chỉ cần nút xác nhận dù lớn. Tối đa 1 ô gõ/dialog, server trả `confirm_text` = mục rủi ro nhất (chốt với design, P010 §9). Model đã train **không khoá mặc định** — người dùng sở hữu dữ liệu của mình.
  2b. **Duyệt theo deployment** (góp ý bao-frontend; **PM chốt 2026-10-12**: mặc định `none` cho bản cài end-user, `threshold` 20 GB cho máy dev dùng chung): cấu hình server `storage.approval_policy`:
      `none` (mặc định cho bản cài cá nhân — người dùng là chủ dữ liệu, chỉ cần dialog) | `threshold` (`approval_threshold_gb`, mặc định 20; dùng cho **máy dev dùng chung của dự án**, khớp D7.4 + skill cleanup-safe-delete).
      Plan JSON trả `approval_policy` + `threshold_gb` (null nếu `none`). Khi `threshold` và plan vượt ngưỡng: `needs_approval=true`, `POST /api/storage/apply` trả 403 với mã catalog **DISK-002 `STORAGE_APPROVAL_REQUIRED`** (severity info; params `{gb, threshold_gb, item_count, item_ids, admin_command}`; action `show_admin_command`; copy `error.DISK-002.*`) trừ khi có `approved_by` hợp lệ (PM/security).
      Body lỗi: `error_code: "DISK-002"`, `error_params{gb, threshold_gb, item_count, item_ids, admin_command}`, `deleted: []`, `approval_request{item_ids, requested_by, requested_at, admin_command}`; admin duyệt bằng CLI `retention.py --items ... --apply --approved-by <tên>` (tính lại plan nên không phụ thuộc plan_id hết hạn). Không bao giờ duyệt từ UI người dùng. Mẫu: `005-annex-storage-plan.threshold.sample.json`. UI hiện banner theo dữ liệu, không hard-code. Policy chỉ đổi được bởi người cài đặt (file config server), không đổi từ UI người dùng.
  3. Mục được bảo vệ (D7 whitelist, LATEST của run đang chạy/interrupted, base model mà adapter đang phụ thuộc, artifact security) → nút bị khoá kèm lý do dễ hiểu.
  4. Thực thi = cùng code path `retention.py --apply` (không có đường xoá thứ hai), ghi event `checkpoint_pruned`/`cache_evicted` với `by=<user>`.
- Gợi ý chủ động: khi preflight (D8) báo thiếu disk → panel đề xuất "Giải phóng X GB từ kết quả tạm" (chỉ cache tính lại được).
- **Phạm vi máy chung giữ nguyên**: panel chỉ thấy/xoá dữ liệu của app. Không hiện nút xoá cho docker image/HF cache ngoài dự án.

### R3. Những gì đổi trong deliverables
- Thêm rule LRU cache vào `retention.py`, đọc/ghi index qua `app.db` (API backend hoặc trực tiếp với WAL + transaction).
- `retention.py` phải có API Python (không chỉ CLI) để backend gọi từ app server:
  `plan(scope, reason, filters) -> Plan` (thuần, chỉ đọc; JSON shape như mục Deliverables) và `apply(plan_id, item_ids, approved_by) -> ApplyResult`.
- HTTP (backend sở hữu, P009 §6, 127.0.0.1): `GET /api/storage/plan?scope=run|cache|all&reason=<error_code>&run_id=` → dry-run JSON + `plan_id` (hết hạn 10 phút);
  `POST /api/storage/apply {plan_id, item_ids[], confirm}` → server **tính lại plan** và chỉ xoá item vẫn còn hợp lệ & nằm trong plan đã xem (chống TOCTOU); item đổi trạng thái → bỏ qua + báo lại.
  Không có endpoint xoá theo path tuỳ ý. Apply đi qua cùng state `evicting→evicted` trong app.db; item thuộc node đang chạy / đang được cache hit bị loại khi tính lại plan (chốt với backend, P009 §6). `confirm` = tên mục khi item > 20 GB hoặc không tính lại được.
- Error catalog action "Giải phóng dung lượng / Xoá checkpoint cũ" (DISK_FULL, preflight exit 3) → mở panel Dung lượng với `reason` + filter sẵn; chỉ gợi ý item an toàn trước (cache tính lại được), theo thứ tự giải phóng ở R1.
- Giai đoạn hiện tại (chỉ reset + setup): không code, không tải gì; chờ Proposal 009 chốt executor rồi viết chi tiết R1.

### Câu hỏi mở thêm
6. ~~(P009) cache executor~~ → đã chốt với backend (R1, P009 §5).
7. (design) Panel "Dung lượng" thuộc màn nào trong app no-code; ngôn ngữ nhãn cho người dùng không biết code.
8. ~~(PM) trần cache 200 GB~~ → đồng ý, kèm 2 điều kiện (R1). ADR-005 cập nhật khi P009 chốt.

## 7. Kiểm tra lại quota theo P001 (điều kiện 2 của ADR-005) — 2026-10-12
Nguồn: P001 §3 (bao-research), `docker/IMAGES.lock`. Ước lượng footprint giai đoạn 2 (chưa có 27B/35B/Omni):
| Hạng mục | Ước lượng | Ghi chú |
|----------|-----------|---------|
| HF models: Qwen3.5 0.8B/2B/4B/9B | ~36 GB | không cái nào > 20 GB |
| Image `nemo-automodel:26.08` (đã pull) | 29.4 GB | chờ probe driver 580.142 |
| Image `vlm-automodel` FROM `nemo-automodel:26.02` | ~25–30 GB giải nén | 13 GB là bản nén |
| Image `vllm/vllm-openai:v0.31.0` | ~22 GB | không dùng chung v0.27.1 (của project khác) |
| Dataset eval regression (không có Video-MME) | ~10–30 GB | P002 |
| Checkpoint LoRA | < 5 GB | vài trăm MB/run |
| Merged 9B + quantized FP8/NVFP4 + engines | ~60 GB | merged 19 GB/bản, giữ ≤ 2 bản |
| **Tổng không tính cache node** | **~200–230 GB** | |
| Cache node (trần) | ≤ 200 GB | evict đầu tiên |
| **Tổng xấu nhất** | **~430 GB** | < warn 500 GB ✅ |
- Kết luận: quota 500/800 GB (user đã xác nhận) **đủ** cho giai đoạn 2. Thêm 1 model 27–35B (55–72 GB) vẫn < 500; thêm ≥ 2 → chạm warn, cần retention/evict cache trước khi tải.
- **Image AutoModel (chốt PM + research)**: chỉ giữ 1 image AutoModel (+1 rollback). Probe 26.08 trên driver 580.142 PASS ⇒ dùng 26.08, **không** pull 26.02. FAIL ⇒ pull 26.02, 26.08 vẫn **giữ** (phương án b' cập nhật DGX OS sẽ làm nó chạy được). 26.08 chỉ vào danh sách ứng viên xoá **sau khi user chọn (a) dài hạn**; khi đó xử lý theo D7 mục 10 (IMAGES.lock → kiểm ancestor → PM xác nhận). Disk còn dư nên giữ không tốn đáng kể.
- Không full-FT 9B: checkpoint có optim ~90 GB/bản (fp32 Adam), và cũng vượt budget RAM.
