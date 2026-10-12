# Proposal 009 — Kiến trúc app node-graph (R-2)
Tác giả: bao-backend (§1–§6, §9) + bao-frontend (§7–§8) · Trạng thái: DRAFT — chờ R-1 (fork/extend) của bao-research · Ngày: 2026-10-12
Liên quan: P008 (runtime: host orchestrator ↔ container worker — trở thành tầng "job" dưới executor), P006 (metrics contract), P007 (memguard), P005 (storage).

## 0. Tóm tắt
App = **graph editor (UI)** + **graph server (backend)**. Backend là nguồn sự thật cho: danh sách node, kiểu port, schema tham số,
kiểm tra graph trước khi chạy, thực thi, cache, resume. UI không hardcode node nào.
```
UI canvas ──REST/WS──► Graph server (FastAPI, host, nhẹ)
                         ├─ Node registry (spec khai báo)
                         ├─ Validator (kiểu, tham số, RAM/disk/thời gian, license)
                         ├─ Executor (topo sort, cache, resume)
                         └─ Job runner (P008) ──docker run──► worker trong container NGC (AutoModel/ModelOpt/vLLM…)
                                      └─ GPU lock + memguard (1 job nặng / lúc)
Artifact store: outputs/runs/<run_id>/ + index SQLite (artifact_id, type, hash, lineage)
```

## 1. Nguyên tắc
1. **Node nặng = 1 job container** (train, quantize, eval, bench, serve). Node nhẹ (đọc config, chọn model, chia data, so sánh báo cáo) chạy in-process trên host.
2. **Artifact có kiểu, bất biến, có hash** — dây nối giữa node truyền *tham chiếu artifact*, không truyền tensor trong RAM (khác ComfyUI: model diffusion load trong process; với ta mỗi bước là container riêng, dữ liệu đi qua disk).
3. **Chặn lỗi trước khi chạy**: mọi thứ kiểm được tĩnh phải kiểm ở validate (người dùng no-tech).
4. **Mặc định an toàn**: tham số mặc định chạy được trên máy hiện tại (ngân sách RAM thật, không phải 128GB).
5. Trung lập với lựa chọn R-1: cùng mô hình này map được vào custom node ComfyUI hoặc server tự xây.

## 2. Node type system
### 2.1 Kiểu port (artifact types)
| Kiểu | Nội dung (metadata trong artifact) | Ví dụ producer → consumer |
|---|---|---|
| `Dataset` (subtype `VLMChatDataset`, `PreferenceDataset`, `EvalSuite`) | path, format, n_samples, splits, license, hash | Load HF dataset → Train |
| `Model` | HF ckpt path/repo@rev, arch, params_b, modalities, license | Pick base model → Train/Quantize/Eval |
| `Adapter` (LoRA) | base_model ref, rank, target_modules | Train(LoRA) → Merge / Eval |
| `Checkpoint` ⊂ `Model` | output full SFT (HF safetensors consolidated), parent run | Train(SFT) → Quantize/Eval/Export |
| `QuantizedModel` ⊂ `Model` | format (fp8/nvfp4/int4_awq), calib info, toolkit version | Quantize → Eval/Bench/Export |
| `Endpoint` | url (127.0.0.1:787x), model ref, runtime (vLLM/TRT-LLM), lifetime | Serve → Eval/Playground |
| `EvalReport` | per-benchmark metrics + predictions.jsonl (P002) | Eval → Compare/Export |
| `BenchReport` | latency/throughput/peak mem (P002 phần B) | Bench → Compare |
| `ExportArtifact` | đóng gói (HF repo dir / GGUF? / TRT engine / NIM) | Export → (sink) |
- "Model gốc" (base) **không phải type riêng** — chỉ là nhãn/chip trên `Model` (`origin: hub|trained|quantized`); khớp R-1 sau khi quy về hệ kiểu này.
- **Tương thích**: subtype nối được vào supertype (`QuantizedModel` → cổng `Model`); `Adapter` + `Model` gộp qua node Merge hoặc cổng `Model|Adapter` của Eval.
- Ràng buộc thêm (kiểm ở validate, không ở kiểu): Adapter.base_model phải khớp Model; QuantizedModel format phải được runtime hỗ trợ (vd NVFP4 Qwen3.5 chưa validated — xem knowledge/backend.md).

### 2.2 Node spec (data) + NodeImpl (hành vi) — tách theo review cleancode
```python
class NodeSpec(BaseModel, frozen=True):      # chỉ data; đây là thứ /api/node-types serialize
    type_id: str                 # "train.lora_vlm"
    version: str                 # semver; đổi hành vi → bump → cache invalid (contract test snapshot schema)
    category: Literal["source", "data", "train", "test", "quantize", "eval", "export", "util"]; icon: str
    # KHÔNG có text cho user — chỉ i18n key suy ra từ type_id:
    #   node.<type_id>.title|desc · node.<type_id>.port.<name> · node.<type_id>.param.<name>.label|help
    inputs: dict[str, PortSpec]  # name -> {type, optional, many}
    outputs: dict[str, PortSpec]
    params_schema: dict          # JSON Schema sinh từ params model (tập con frontend: string, number, int, bool, enum, array-of-enum, file-ref);
                                 # widget/order/advanced/preset đặt NGAY trên field: Field(json_schema_extra={"ui": {...}}) — 1 chỗ khai báo
    cacheable: bool = True
    runner: Literal["host", "container"]
    image: ImageRef | None
    expand_to: SubgraphRef | None = None   # RESERVED (không làm v1, P010)

class NodeImpl(Protocol):
    params_model: type[BaseModel]          # = config pydantic của engine (hoặc subset), không định nghĩa lại
    def estimate(self, params, inputs_meta) -> Estimate: ...
    # Estimate = {ram_gb:[lo,hi], disk_gb:[lo,hi], minutes:[lo,hi], gpu_exclusive, source: measured|formula|declared|unknown, basis}  # declared = tác giả plugin khai, chưa đo
    def check(self, params, inputs_meta) -> list[Issue]: ...    # lỗi nghiệp vụ: chỉ code + params
    def build_job(self, params, inputs, run_dir) -> JobSpec: ... # → job runner P008
```
- Registry = list tường minh các cặp (spec, impl) trong `nodes/__init__.py` (P004 §8.1). **Không** entry-point plugin ở v1 (YAGNI + nạp code tuỳ ý là rủi ro security).
- Contract test: snapshot hash `params_schema` từng node; schema đổi mà `version` không đổi → test fail (chống cache stale).
- **Domain object không mang text**: `Issue`/`UserError` = {code, severity, node_id, port, param, params (placeholder), details đã redact}; không hậu tố `_vi`.
  Text resolve ở rìa (app/API) bằng `i18n.render(code, params, locale)`. Test: domain chỉ assert code + params; bộ test i18n riêng trên vi/en.yaml.
- **i18n — 1 nguồn duy nhất** (chốt design + frontend + fixcode): `src/vlm_pipeline/i18n/{vi,en}.yaml` (bao-design sở hữu). Lỗi: `error.<code>.title|what|why`, `action.<id>`, `consequence.<code>`.
  `GET /api/node-types?locale=vi` và response validate trả text đã resolve **kèm key gốc**; thiếu key → trả chính key (không crash). CI test: mọi key được dùng có đủ vi + en.
- Mã lỗi theo catalog của fixcode (MEM-001, DISK-001, CFG-001…); `error_catalog.yaml` chỉ giữ {code, match, action, key, params}.
- Category `source` = làn "Nguồn" (dữ liệu + model gốc): `model.hf_pick`, `data.hf_load`, `data.local_folder`.
- **Registry PortType** (`GET /api/node-types` trả kèm): `{id, parent, color_token, label_key}` — UI tô màu/kiểm tra nối dây theo `parent` (subtype).
- Node nặng chỉ là **adapter mỏng** dịch params → recipe chính chủ (AutoModel YAML, `hf_ptq.py` args, `vllm serve` args) — đúng P004/P008.

### 2.4 Node bên thứ 3 (plugin) — v1 theo ADR-009, sandbox theo P003 D16, layout P004 §8.5
- **Đóng gói**: node pack = manifest khai báo (tên file chốt 1 cái với security/cleancode: đề xuất `vlm-node.toml` của D16) + source → build thành **1 image riêng** trong sandbox build, pin commit SHA 40-hex + sha256 tarball + digest image.
- **Loader**: dựng `NodeSpec` từ manifest (data), **không import code plugin vào server**; manifest extra=forbid, không JS/CSS, UI chỉ từ JSON Schema + `x-ui` + i18n key. Registry = list core tường minh + plugin manifest hợp lệ đã cài.
- **Chạy**: CHỈ container (không venv/subprocess host), qua đúng Worker/job runner P008 với profile sandbox D16: uid 61001, rootfs ro, `/in` ro + `/out`, network none (v1), không secret/env host, docker args do orchestrator tự sinh từ bảng quyền. Output validate (magic bytes chặn pickle, loại file whitelist safetensors/json/jsonl/parquet/ảnh/video, size) **trước** khi chuyển vào vùng artifact; duyệt `/out` bằng lstat/`O_NOFOLLOW`, không follow symlink. `--pids-limit 4096`, `--read-only` + tmpfs `/tmp` giới hạn size. Manifest khai network ≠ none → từ chối cài (mã catalog fixcode cấp cho `permission_unsupported`). v1 chỉ build image plugin từ source FROM base `vlm-*` trong allowlist, không nhận image dựng sẵn.
- `plugin_api` (phía worker, semver): đọc `/in` + resolved config, ghi `/out` + events/metrics theo contract, raise lỗi bằng code catalog.
- **NodeSpec thêm** (frontend F6d + design P010 §15):
  `origin {kind: first_party|third_party, publisher, pack, version, commit_sha, license, installed_at, reviewed: bool}`,
  `permissions` (đã chuẩn hoá bởi orchestrator: {gpu, network: [], max_runtime, max_memory_gb} — UI không hiện text tác giả viết),
  `policy_block {code, reason_key} | None` (vd node bên thứ 3 gpu=true bị chặn trên host chưa vá toolkit).
- Số `declared` của plugin là **untrusted**: chỉ hiển thị (nhãn "tác giả node tự khai, chưa đo"), không bao giờ dùng để nới preflight; preflight dùng max(declared, ước lượng của ta); giới hạn chạy do orchestrator + memguard quyết định.
- **Workflow có node chưa cài**: validate trả `CFG-004` params `{node_type, reason: not_installed|not_allowed}`; lệch version → `CFG-009` NODE_VERSION_MISMATCH `{node_id, node_type, workflow_version, installed_version, migratable}` (actions `migrate_node` khi migratable, `open_node`). Server giữ nguyên node + dây (UI placeholder), **không bao giờ tự cài**.
- **API plugin** (thêm vào §6):
  `POST /api/plugins/install/plan {source: git_url, commit_sha}` → dry-run: manifest, quyền chuẩn hoá, kết quả quét tĩnh, license, deps, kích thước image, sha256 — chưa cài gì, trả `plan_id`;
  `POST /api/plugins/install/apply {plan_id, ack}` (quyền Cao cần `ack.typed_name`); `GET /api/plugins`;
  `GET /api/plugins/{id}/usages` → workflow đang dùng; `DELETE /api/plugins/{id}` (xoá cả image);
  `POST /api/plugins/{id}/update/plan {commit_sha}` → diff quyền (added/removed) + diff code (file/dòng); quyền tăng → apply phải ack lại từ đầu.

#### 2.4.1 Chốt bổ sung (security D16 + design P010 §15 + frontend F6d/F6e + fixcode)
- `trust_level: first_party|reviewed|unverified` (enum chung). **Mọi** mức cảnh báo do server tính theo trust_level × quyền, UI chỉ hiển thị.
- `permissions[]` mỗi phần tử: `{id: gpu|network|runtime|memory, level: info|warning|critical, requires_typed_confirm, perm_copy_key, confirm_copy_key}` (vd GPU + unverified = critical/true; GPU + reviewed/first_party = warning/false; network = critical/true, v1.1).
- Install plan trả thêm `requires_typed_confirm` (tổng hợp) và `sandbox_verified: bool` — chỉ true khi build metadata ghi test D16 PASS **và** lúc khởi động orchestrator tự kiểm docker args (non-root, cap_drop, không mount secret); không bật được bằng config.
- Endpoint cài chỉ nhận: (a) `registry_id` từ registry/allowlist của app, hoặc (b) `{repo_url, commit_sha: ^[0-9a-f]{40}$, user_supplied: true}` (branch/tag/SHA ngắn → từ chối). **Không** có cài theo workflow, **không** cài hàng loạt. Schema workflow extra=forbid, không có field nguồn cài (`install_url`/`source` → CFG-007).
- Node trong workflow có thể mang `package` (tuỳ chọn) — **chỉ là khoá tra registry**, không bao giờ echo lại trong response (kể cả technical_detail).
- `CFG-004` params: `{node_type, reason: not_installed|not_allowed, trust_level: first_party|reviewed|unverified (server tính từ registry/allowlist + hash, không lấy từ workflow; không tra được → unverified), name_lookalike_of: type_id|null, suggested_package: chỉ lấy từ registry/allowlist|null}`. `name_lookalike_of` khi tên trùng/gần giống node first_party/reviewed (cùng tên khác publisher, hoặc dùng prefix `vlm/` dành riêng).
- Preflight với `source=declared`: cộng biên an toàn rộng hơn (design), vẫn max(declared, ước lượng ta).

### 2.3 Bộ node v1 (đề xuất, bám pipeline)
Data: `data.hf_load`, `data.local_folder`, `data.format_vlm_chat`, `data.split`, `data.preview` ·
Model: `model.hf_pick` (danh mục model đã vet bởi security) ·
Train: `train.sft_vlm`, `train.lora_vlm` (AutoModel), `train.grpo_vlm` (NeMo-RL, sau) ·
Test: `serve.vllm` → `playground.chat` · Quantize: `quantize.modelopt_ptq` · Eval: `eval.benchmarks` · Bench: `bench.perf` ·
Export: `export.hf_dir`, `export.merge_lora`, `export.trtllm` · Utility: `compare.reports`, `note`.

## 3. Workflow file (lưu/chia sẻ, template)
```json
{"format":"vlm-workflow","schema_version":1,"name":"Fine-tune VLM bằng LoRA",
 "nodes":[{"id":"n1","type":"model.hf_pick","version":"1.0.0","params":{"repo":"Qwen/Qwen3.5-2B","revision":"<sha>"},"ui":{"x":0,"y":0}}, ...],
 "edges":[{"from":["n1","model"],"to":["n3","base_model"]}, ...],
 "requires":{"node_types":{"train.lora_vlm":">=1.0"}}}
```
- `ui` (position, màu, collapsed) **không** đi vào cache key — kéo node không làm job chạy lại; cache key chỉ từ `type, version, params` + input.
- Template = workflow file trong `templates/` + `README`/thumbnail; mở ra là chạy được với tham số mặc định an toàn.
  Template mặc định chỉ dùng model/dataset **APPROVED không điều kiện** của security (MedPix-VQA là CONDITIONAL → chỉ spike, không vào template).
- Migration: `schema_version` + `version` từng node; server có hàm nâng cấp workflow cũ (như ComfyUI xử lý node cũ).
- Không lưu secret trong workflow; tham chiếu `secret://HF_TOKEN`.
- **Tham chiếu tài nguyên** (`common/config`, chốt với security + tester), đều frozen, extra=forbid:
  - `AssetRef(repo_id, revision: ^[0-9a-f]{40}$, repo_type: model|dataset, split?)` — không nhận branch/tag.
  - `ImageRef(name: chỉ nvcr.io/nvidia/* hoặc allowlist, digest: ^sha256:[0-9a-f]{64}$)` — image không đi qua AssetRef.
  - `LocalRef(path: relpath trong workspace, manifest_sha256)` — kiểm realpath + prefix workspace **ngay lúc mở** (chống TOCTOU symlink), verify manifest trước khi đọc.
  - Load model: `use_safetensors=True`; `trust_remote_code` chỉ theo allowlist (repo_id, revision) **kể cả repo bắc cầu qua auto_map** (vd Nemotron-Omni → C-RADIOv4-H).

## 4. Validation trước khi chạy (`POST /api/graphs/validate`)
Response: `{errors:[{node_id, port, param, code, severity: error|warn, params, title_key, title, what, why, action_ids}], estimate:{per_node:{...}, ram_gb_peak, disk_gb, duration_min, gpu_exclusive}}`. Mã: cấu trúc/kiểu/tham số = CFG-001..008 (CFG-008 NODE_ID_DUPLICATE), path/secret = SEC-001/002, license = SEC-003, RAM = MEM-001, disk = DISK-001. Thứ tự kiểm:
1. Cấu trúc: chu trình, cổng bắt buộc chưa nối, node type/version không tồn tại.
2. Kiểu: dây nối kiểu không tương thích.
3. Tham số: JSON Schema + `NodeSpec.validate` (vd LoRA rank, max_steps, image size).
4. Ngữ nghĩa xuyên node: Adapter chỉ gắn Model có cùng `base_repo@revision` (LoRA 2B không nối vào 9B — research); Quantize nhận Model (Adapter phải merge trước); format quantize được runtime hỗ trợ; dataset có split eval.
5. Tài nguyên: `estimate()` từng node vs `preflight` (P007): **RAM đỉnh từng node nặng** so với MemAvailable − margin; tổng disk; thời gian ước lượng. Trả mức "đỏ/vàng/xanh" cho UI.
6. Chính sách: license model/dataset (configs/security/licenses.yaml), download > 20GB cần xác nhận.
Ước lượng RAM ban đầu = bảng số đo thật (spike, P1-9) + công thức (params × bytes/param × hệ số LoRA/full, + activations theo seq/ảnh); hiệu chỉnh dần từ run thật (peak_mem_gb trong run.json).

## 5. Executor
- **Lập lịch**: topo sort; node host chạy song song (thread pool nhỏ); node container vào **1 hàng đợi GPU** (FIFO, GPU_LOCK của P008). Node `serve` giữ GPU → coi như job dài, các node phụ thuộc Endpoint chạy trong khi serve sống, xong thì executor tự tắt serve.
- **Cache (như ComfyUI)**: `cache_key = H(type_id, version, params_canonical, [input artifact hash], image_digest)`. Nếu artifact có cache_key đó đã `completed` → tái dùng, không chạy lại. Sửa 1 node → chỉ node đó và hạ nguồn chạy lại.
  Artifact hash: dataset/model = hash manifest file (tên+size+sha của shard, tính 1 lần, lưu index), không hash lại GB dữ liệu mỗi lần.
- **Cache trên disk, không trong RAM** (bao-cleanup, khớp ADR-005 / P005 §6 — output là GB, RAM cache dễ OOM trên GB10):
  - Dữ liệu node nặng nằm ở **một chỗ duy nhất**: `outputs/runs/<run_id>/`. **SQLite `outputs/app.db` là index duy nhất** cho cache/artifact
    (cache_key, run_id, state, last_hit_at, pinned, evicted_at); không có entry.json song song (nếu cần chỉ là bản export read-only). Bật WAL; retention.py dùng API Python chung.
  - Node tải HF (`model.hf_pick`, `data.hf_load`) → artifact tham chiếu HF cache dự án (`.cache/hf`), không lưu bản thứ hai. RAM cache chỉ cho output nhỏ ≤ 2GB.
  - `cacheable=false`: `serve.vllm`/`Endpoint`, `playground.chat`, `export.*`, nguồn local chưa có manifest hash.
  - **Evict ≠ xoá run dir**: chỉ xoá bytes nặng (checkpoints/, engines/, output nặng; không đụng export/ đã đặt tên); giữ run.json, *.jsonl, logs/ (whitelist D7) → run.json `status=evicted` + event `cache_evicted`.
  - Evict 3 bước chống race: (1) transaction đặt `state=evicting` (executor coi là miss) → (2) xoá file → (3) `state=evicted`.
    Cache hit: executor kiểm path tồn tại + hash manifest; lệch → miss + cảnh báo.
  - Lineage: cạnh có cờ `requires_parent_at_load` (vd Adapter cần base là output full-FT của node trước) → cha không được evict khi con còn sống. Quantized/merged tự chứa → cha evict được.
  - Trần LRU 200GB chỉ tính run evict được (không pinned/KEEP/export); chạm quota 500GB → thứ tự evict theo ADR-005 (cache trước, checkpoint D3 sau).
  - `cache_key` công khai, hàm thuần `compute_cache_key(spec, params, input_hashes, image_digest) -> str` (test được — bao-tester).
- **Testability** (bao-tester): `validate(graph) -> list[Issue]` là hàm thuần tách khỏi executor; executor nhận `Worker` protocol
  (`run(node, inputs, run_dir) -> Result`) → test bằng `FakeWorker` trên CPU, inject lỗi theo lớp; trạng thái node persist ra disk (SQLite) để resume;
  mọi exception từ worker → error code trong catalog, fallback `SYS-000`. Catalog: `.team/knowledge/error-catalog-draft.md` (fixcode sở hữu; mã mới phải xin fixcode, không tự đặt).
- **Hiển thị cache/resume (P010)**: event `node_status` khi cache hit kèm `cached_from{run_id, completed_at}` → UI "↺ dùng lại kết quả hh:mm";
  node `interrupted`/`failed` có `resumable: bool` + `checkpoint{step, saved_at}` → nút "Tiếp tục từ checkpoint".
- **Resume**: run graph có `graph_run_id`; mỗi node có trạng thái `pending|queued|running|completed|failed|interrupted|cached|skipped`. Chạy lại graph = các node completed được cache; node train `interrupted` (memguard exit 75) resume từ checkpoint LATEST.
- **Phân loại exit của worker** (fixcode, theo thứ tự): (0) executor đã chủ động huỷ (user Huỷ/timeout → SIGTERM → SIGKILL sau grace) → exit 137/143 **không phải lỗi**: `cancelled`/`interrupted` (resumable nếu có checkpoint), không gán mã; (1) exit 75 → MEM-002 `interrupted`; (2) exit 137/SIGKILL **và** (chỉ khi có bằng chứng, không đoán) (`.State.OOMKilled=true` hoặc `oom_kill` tăng trong cgroup `memory.events`) → MEM-003;
  (3) có event `error` trong events.jsonl → mã của event; (4) còn lại (139 SIGSEGV, 134 SIGABRT…) → SYS-003 `WORKER_CRASHED` {exit_code, signal} + 30 dòng log cuối trong technical_detail; exception không phân loại được → SYS-000.
  Vì cần đọc `OOMKilled`, container **không chạy `--rm`**: runner `docker inspect` lấy state + exit_code (ghi vào run.json) rồi mới `docker rm`.
  Cache hit nhưng artifact mất/manifest lệch → CACHE-001 (warn, chạy lại, không chặn).
- **Huỷ**: cancel node → SIGTERM container (grace để save ckpt) → `interrupted`.
- Mọi event lỗi runtime mang `resumable: bool` + `checkpoint_step?` lấy từ checkpoint **thật** trên disk (UI chỉ hiện "Tiếp tục từ chỗ đã lưu" khi đúng).
- **Shape lỗi (chốt cleancode + fixcode)**: catalog KHÔNG chứa text; domain exception, `Issue` của validate và event `error` chỉ mang `code + params` (+ node_id, port, technical_detail đã redact). Text từ `i18n/{vi,en}.yaml`, render ở rìa (server response/UI).
- **Lỗi dễ hiểu**: worker ghi `events.jsonl` loại `error` có `code` theo catalog fixcode (MEM-003 UMA OOM, NET-001 tải lỗi `reason: gated|auth|network`, DISK-001…; fallback SYS-000); server map code → key i18n `error.<code>.*` + action; stack trace chỉ ở "chi tiết".
- Persist: SQLite (`outputs/app.db`): graph_runs, node_runs, artifacts(cache_key, type, path, hash, parent ids). Thư mục run vẫn là `outputs/runs/<run_id>/` (P005/P006) — mỗi node nặng 1 run dir, `parent_run_id` → lineage.

## 6. API backend (FastAPI, bind 127.0.0.1:7870 cùng dashboard — P006)
| Method | Path | Mục đích |
|---|---|---|
| GET | `/api/node-types` | danh sách NodeSpec + JSON Schema params + kiểu port |
| GET | `/api/templates`, `/api/templates/{id}` | template gallery |
| POST | `/api/graphs/validate` | §4 |
| POST | `/api/runs` | chạy graph (body = workflow) → graph_run_id |
| GET | `/api/runs/{id}` | trạng thái node, artifacts |
| POST | `/api/runs/{id}/cancel` · `/resume` | |
| WS | `/api/runs/{id}/events` | stream events/metrics (cùng schema events.jsonl/metrics.jsonl) + event `node_status{node_id, status: queued|cached|running|done|failed|skipped|interrupted, progress, eta_s}` |
| GET | `/api/artifacts/{id}` (+ `/preview`) | xem dataset sample, report, so sánh |
| GET | `/api/system` | RAM/disk/GPU lock (cho header) |
| GET | `/api/storage/plan` | dry-run dọn dẹp (gọi `retention.plan()` của cleanup, P005 §6 R3) → JSON items + `plan_id` (hết hạn 10 phút) |
| POST | `/api/storage/apply` | `{plan_id, item_ids[], confirm}` → server tính lại plan, chỉ xoá item còn hợp lệ VÀ nằm trong plan đã xem (chống TOCTOU); dùng cùng cơ chế `evicting→evicted` của §5 nên không đua với executor. **Không có endpoint xoá theo path tuỳ ý.** |
|  | (nhánh duyệt, P005 §6 R2 2b) | Config server `storage.approval_policy = none | threshold` (`approval_threshold_gb`, mặc định 20; máy dùng chung = threshold). Vượt ngưỡng → **403** `{code: "DISK-002", params {gb, threshold_gb, item_count, item_ids, admin_command}, deleted: [], approval_request{item_ids, requested_by, requested_at, admin_command}}`; admin duyệt bằng `retention.py --items … --apply --approved-by <tên>` (tính lại plan). Plan JSON có `approval_policy`, `threshold_gb`. |

## 7–8. Canvas, node library, inspector, template gallery, live run view — xem phụ lục của bao-frontend
→ [.team/proposals/009-annex-frontend-canvas.md](009-annex-frontend-canvas.md) (khuyến nghị React Flow `@xyflow/react` 12.12, MIT; form engine theo tập con JSON Schema ở §2.2).

## 9. Phương án nền tảng (giữ mở tới khi R-1 về — PM yêu cầu trình bày công bằng)
Effort = **ước lượng thô người-tuần, chưa đo** cho MVP (canvas + 6–8 node + validate + executor + 2 template), chưa tính engine NVIDIA (chung mọi phương án).
| | (1) Tự xây: FastAPI + React Flow | (2) Tự xây backend + **LiteGraph** (lõi canvas ComfyUI; bản MIT đã ngừng) | (3) **ComfyUI process riêng** + custom nodes gọi engine của ta qua API | (4) Nền NVIDIA (AI Workbench / NeMo Customizer) |
|---|---|---|---|---|
| Effort MVP | ~6–8: executor ~1–2 + UI ~4–5 (= editor ~2: canvas, typed port, inspector, undo — số trong phụ lục frontend; + ~2–3: template gallery, validation overlay, live run view, panel P006) | ~7–10: như (1) nhưng UI +1–2 tuần (widget vẽ trên canvas, inspector/form React làm overlay ngoài canvas, thêm công a11y + lỗi tiếng Việt trên node) | ~3–4 cho bản chạy được; +3–5 để đạt validate/estimate/lỗi tiếng Việt | chưa rõ — R-1 |
| Được | Toàn quyền UX no-tech (form tự sinh, lỗi tiếng Việt, estimate RAM), khớp P006/P008, license sạch (MIT/Apache) | UX quen với user ComfyUI, canvas hiệu năng cao | Canvas, queue, cache, lưu workflow có sẵn; người dùng ComfyUI quen; hệ sinh thái custom node | Hỗ trợ chính chủ, tích hợp NGC |
| Mất / rủi ro | Tự viết + bảo trì editor; không có hệ sinh thái custom node | Bản gốc (jagenjo) ngừng từ 2024; `@comfyorg/litegraph` đã **DEPRECATED** (release cuối 0.17.2, 2025-08-06) — chỉ còn bản nằm trong ComfyUI_frontend (GPL); khó tích hợp React UI (inspector/form); vẫn phải tự viết mọi thứ backend | License (security, P003 D6b): **khả thi** nếu ranh giới cứng — ComfyUI không sửa, process riêng, kèm source/offer GPL; custom node (import `comfy.*`) phải GPL-3.0 trong repo/package riêng, chỉ là lớp mỏng gọi orchestrator qua HTTP/JSON; orchestrator + engine giữ Apache-2.0, không import chéo. Chi phí thật: pin ComfyUI, không dùng Manager, tự theo dõi CVE; executor in-process, kiểu port là string tự do, hàng đợi 1 prompt, không có validate tài nguyên trước chạy; UI dày đặc khó đơn giản hoá cho no-tech; phụ thuộc nhịp đổi API của ComfyUI; 2 design system song song (PrimeVue/Tailwind của Comfy vs tokens của ta); panel P006 phải nhúng qua extension (phụ lục F1b) | Có thể không chạy local/arm64, khó tuỳ biến node-graph; lock-in |
| Hệ sinh thái custom node diffusion/ComfyUI | không dùng được | không dùng được (chỉ canvas) | dùng được, nhưng phần lớn node là diffusion, ít giá trị cho training VLM | — |
| Ảnh hưởng backend (§2–§6) | dùng nguyên | dùng nguyên | NodeSpec → class node ComfyUI; validate/estimate phải làm thành pre-check riêng hoặc patch frontend | thay phần lớn |
Nhận định backend: (1) phù hợp nhất với yêu cầu "chặn lỗi trước khi chạy" cho người no-tech, vì validate/estimate/resume là phần khó và không phương án nào có sẵn;
executor của ta đơn giản vì mỗi node là job container. (3) là đường nhanh nhất ra demo; license khả thi với ranh giới HTTP cứng (security đã xác nhận), cái giá còn lại là bảo trì/bảo mật + UX. Chốt sau R-1.
## 9b. Phạm vi v1 vs v2 (review cleancode #6 — gọn MVP)
| v1 (MVP) | v2 (khi có số đo thật) |
|---|---|
| Executor **tuần tự** (1 node/lúc, cả host lẫn container) | thread pool cho node host song song |
| Cache hit + resume ckpt + kiểm manifest khi hit | LRU tự động 200GB |
| Evict **thủ công** qua `/api/storage/plan|apply`; `requires_parent_at_load` chỉ là field dữ liệu được plan tôn trọng | evict tự động theo quota |
| `schema_version` trong workflow (ghi + từ chối version lạ bằng CFG-007) | hàm migrate workflow cũ |
| SQLite 1 writer (executor), WAL | — |

## 9c. Layering (P004 §8.1)
`app | cli` > `graph` (workflow model, validate, cache_key, executor) > `nodes` > { `engines` (adapter AutoModel/ModelOpt/vLLM), `artifacts` (index SQLite + API retention dùng chung app/storage & executor) } > `common`.
`engines` và `artifacts` là **anh em, không import nhau** (engine chỉ nhận path, không biết index). i18n = package data.

## 10. Ảnh hưởng tới proposal khác
- P008: giữ nguyên, trở thành lớp "job runner" dưới executor; CLI `vlm <stage>` vẫn giữ cho dev/test (cùng code adapter của node).
- P006: dashboard trở thành 1 phần của app (cùng server/port); metrics contract giữ.
- P005: thêm `outputs/app.db` + cache artifacts theo cache_key — retention không xoá artifact đang được cache hit gần đây / thuộc graph được pin.
- P002/tester: mỗi node có test spec (params mặc định + tiny model) → smoke test template.

## 11. Câu hỏi mở
1. R-1: ComfyUI fork hay tự xây? (quyết định §9)
2. Node `serve` sống lâu có giữ GPU lock không, hay cho phép eval song song cùng container? (đề xuất: giữ lock, eval gọi endpoint trong cùng "phiên GPU")
3. Phạm vi v1: chỉ 1 máy, 1 người dùng (không auth) — đồng ý?
