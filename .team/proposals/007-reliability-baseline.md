# Proposal 007 — Reliability baseline: preflight, UMA memory guard, env fingerprint, conversion round-trip, bug workflow

- Tác giả: bao-fixcode · Ngày: 2026-10-12 · Trạng thái: **DRAFT — chờ review**
- Reviewer cần: bao-backend (tích hợp vào run.py/worker.py), bao-cleancode (khớp layout 004: `common/memguard.py`, `common/gpu_lock.py`), bao-tester (test + bug loop), bao-security (không lộ secret trong fingerprint), bao-pm (chốt ADR)
- Phụ thuộc: 001 (container/version), 004 (layout). Nền tảng: `.team/knowledge/fixcode.md`

## 1. Vấn đề
Máy là GB10 dùng chung, UMA 128GB nhưng chỉ còn ~43GB `MemAvailable` (đo 16:01). Ba lớp lỗi đắt nhất trên stack này (KB §2):
1. **Memory UMA**: `nvidia-smi` không báo memory, `torch.cuda.mem_get_info` báo thiếu, và khi cạn bộ nhớ host có thể **treo cả máy** thay vì ném OOM (ảnh hưởng kafka/elastic/banking-rag của người khác).
2. **ABI/version drift** (CUDA12 wheel, sm_121a/ptxas, flash-attn) → lỗi chỉ lộ khi đã chạy GPU.
3. **Convert/export sai im lặng** (HF⇄Megatron, LoRA merge, ModelOpt export): load được nhưng output sai → mất cả vòng benchmark.
Không có cơ chế chung thì mỗi stage tự xử lý (hoặc không), và bug report thiếu dữ kiện để repro.

## 2. Đề xuất
| # | Đề xuất | Vị trí (theo 004) |
|---|---|---|
| R1 | **Preflight** chạy trước mọi job nặng: arch=aarch64, `torch.version.cuda` 13.x, capability (12,1), torch không phải `+cpu`, `/dev/shm` ≥ 8GB (hoặc `--ipc=host`), GPU_LOCK rảnh, memory budget đủ (R2). Fail → thông báo rõ nguyên nhân + KB section, không chạy. | `common/preflight.py` (host-safe phần host; phần torch gọi trong worker) |
| R2 | **Memory budget** = `MemAvailable` (KHÔNG cộng swap) − margin, margin = `max(10GB, 20% × expected_peak_gb)` (P005 D8 + PM đề xuất 2026-10-12). Thêm điều kiện `SwapFree ≥ 2GB`. Mỗi stage config khai `expected_peak_gb`; preflight từ chối nếu budget < peak. **Một code path duy nhất**: `scripts/cleanup/preflight.py` (P005) gọi `common/memguard.check_budget()` rồi thêm phần disk/lock — ngưỡng RAM chỉ nằm ở memguard. | `common/memguard.py` |
| R3 | **Memory guard runtime**: thread đọc `/proc/meminfo` mỗi 2s trong worker; `MemAvailable < warn_gb (14)` → log cảnh báo + metric; `< abort_gb (8)` → set cờ dừng để trainer checkpoint & thoát có kiểm soát (exit code riêng `75`, run.json `status=interrupted` — retention coi như failed/running, không strip optimizer để còn resume); nếu 30s không thoát → `os._exit` chính process của mình. Ghi min `MemAvailable` + `torch.cuda.max_memory_allocated` vào run.json. Không bao giờ đụng process khác. | `common/memguard.py` + hook trong `*/worker.py` |
| R4 | **Env fingerprint** tự động ghi vào `outputs/<run_id>/env.json` (và tóm tắt trong run.json): git sha + dirty, container image + digest, driver, versions (torch/cuda/cudnn/transformers/TE/megatron-bridge/automodel/modelopt/vllm/trtllm/triton), env vars theo **allowlist tên cụ thể** (không dùng prefix/denylist — theo bao-security), value khớp `://…@`, `hf_`, `nvapi-`, `sk-`, `token|key|secret|pass|auth|cred|cookie|session` → `<redacted>`; `HF_TOKEN` không bao giờ ghi kể cả hash. File được test secret-scan của tester quét, MemAvailable lúc start. Cùng nội dung với `fixcode-triage/scripts/env_fingerprint.sh`. | `common/runinfo.py` (field `env`) |
| R5 | **Defaults an toàn cho GB10** đặt trong worker (có thể override, phải ghi lý do): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `attn_implementation="sdpa"` (đến khi 001 xác nhận flash-attn/TE FA cho sm_121), `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` nếu tồn tại, vLLM `gpu_memory_utilization ≤ 0.25` + `max_num_seqs ≤ 4`, docker `--ipc=host`. Mỗi default có comment `WORKAROUND(...)`/link KB. | `*/worker.py`, `docker/` |
| R6 | **Test tính đúng của biến đổi weights** (G0, chạy trước gate chất lượng P002 G1–G4; ngưỡng theo review bao-tester): (a) HF→Megatron→HF cùng dtype: **bit-exact** (`torch.equal`, đủ key, không key thừa). (b) Logits: cosine ≥0.999 hoặc mean KL ≤1e-3; `max_abs_diff` chỉ log; cùng kernel + cùng `attn_implementation`. (c) Tiny-random (CPU, `tests/integration/`): bit-exact weights + logits fp32 allclose; KHÔNG dùng top-1. (d) Model thật (`tests/gpu/`, GPU lock): greedy ~16 prompt **có ảnh**, ngắn+dài; pass khi khớp hoàn toàn hoặc chỗ lệch đầu tiên là near-tie (top1−top2 < 0.1 logit). (e) LoRA merge: W_merged vs tham chiếu fp32 `W + scale·B·A` (tol tương đối ~1e-2 bf16); logits merged vs unmerged theo (b)+(d). (f) Quantize: assert precision thực tế của vision tower/projector **khớp `vision_precision` trong config** (mặc định bf16); top-1 agreement chỉ log, không gate; mini-eval 50 mẫu chỉ là smoke (score ≥ base−10pt, output không rỗng/rác). | `tests/integration/test_*_roundtrip.py`, `tests/gpu/` |
| R7 | **Bug workflow** (đã thống nhất với bao-tester/cleancode): `.team/bugs/BUG-NNN-<slug>.md`, status OPEN→FIXING→FIXED-UNVERIFIED→VERIFIED; env = output R4/script; test regression mang ID bug; commit `fix(<module>): ...` có `Root cause:`; workaround theo chuẩn `# WORKAROUND(BUG-NNN): <why> — remove when <cond>`; bug class mới → KB §3. | `.team/bugs/`, skill `fixcode-triage` |
| R8 | **Debug artifacts** vào `outputs/debug/BUG-NNN/` (env, log, memory snapshot `torch.cuda.memory._dump_snapshot`) + file `STATUS` 1 dòng `OPEN|FIXING|FIXED|VERIFIED YYYY-MM-DD`. Snapshot: tối đa 1 file/bug (ghi đè), `max_entries` ≤ 100k, cap 1GB; pickle chỉ mở bằng memory_viz/của chính ta, không load pickle từ nguồn lạ. Retention: xoá được sau VERIFIED + 14 ngày (đã có trong P005). | 005 |

## 3. Interface phác thảo (không phải code cuối) — đã thống nhất với bao-cleanup + bao-backend, review bởi bao-cleancode
`common/memguard.py` là **nơi duy nhất** chứa logic preflight + watchdog (stdlib only, host-safe, mypy strict). `scripts/cleanup/preflight.py` (P005) và job runner của backend chỉ là wrapper mỏng. Phần disk check do bao-cleanup PR vào.
```python
EXIT_OK, EXIT_NO_MEM, EXIT_NO_DISK, EXIT_LOCK_BUSY = 0, 2, 3, 4   # preflight CLI exit codes
EXIT_MEM_PREEMPT = 75                                             # worker stopped by watchdog → run.json status=interrupted

@dataclass(frozen=True)
class MemSnapshot: mem_available_gb: float; swap_free_gb: float; cached_gb: float

def read_meminfo(path: Path = Path("/proc/meminfo")) -> MemSnapshot: ...
def read_cgroup_limit_gb(root: Path = Path("/sys/fs/cgroup")) -> float | None: ...   # cgroup v2 memory.max - memory.current; separate fn
def effective_available_gb(snap: MemSnapshot, cgroup_gb: float | None) -> float: ...  # min of both

@dataclass(frozen=True)
class Verdict: ok: bool; exit_code: int; reasons: list[str]; snapshot: MemSnapshot; required_gb: float  # printed as JSON on stdout

def check_budget(expected_peak_gb: float, *, snapshot: MemSnapshot | None = None,
                 min_margin_gb: float = 10.0, margin_frac: float = 0.2, min_swap_free_gb: float = 2.0) -> Verdict: ...
def preflight(expected_peak_gb: float, *, snapshot: MemSnapshot | None = None, ...) -> Verdict: ...  # RAM + swap + disk + lock

class MemGuard:  # context manager, background thread
    def __init__(self, *, warn_gb: float = 14, abort_gb: float = 8, interval_s: float = 2,
                 meminfo_path: Path = Path("/proc/meminfo"),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.should_stop: threading.Event = threading.Event()   # per-instance, polled by trainer each step
        self.min_available_gb: float = math.inf
    def poll_once(self) -> None: ...  # testable without threads/sleep
```
`InsufficientMemoryError(VlmPipelineError)` trong `common/errors.py`. Unit test: meminfo/cgroup fixture files, `poll_once` với clock giả → không cần GPU, không sleep thật.
`common/` thuộc backend → trước khi code tôi báo "đang giữ" với backend + cleancode.

Lưu ý container: `/proc/meminfo` trong docker (không lxcfs) hiển thị số liệu **host** — đúng thứ ta cần trên UMA. Nếu dùng `--memory` cgroup thì `effective_available_gb` lấy min.

## 4. Không chọn
- **Dựa vào `docker --memory` làm bảo vệ duy nhất**: forum NVIDIA báo vẫn treo host trong container `--memory=100g` (KB §2.2). Giữ làm lớp phụ.
- **earlyoom/sysctl/drop_caches tự động**: cần sudo, thay đổi host chung → ngoài quyền; nếu muốn, PM hỏi user.
- **`torch.cuda.set_per_process_memory_fraction`** làm cơ chế chính: hành vi trên UMA chưa xác minh; tôi sẽ thử nghiệm nhỏ rồi báo lại.
- **Cộng swap vào budget**: swap đang dùng 8/15GB; chạm swap là hiệu năng sập + rủi ro treo.

## 5. Kế hoạch
1. Sau ADR 004/007: tôi viết `common/memguard.py` + `common/preflight.py` + unit test (CPU-only) — PR nhỏ, bao-cleancode review. Không chạm file khác khi chưa "giữ" với cleancode.
2. bao-backend gắn hook MemGuard + preflight + env.json vào `run.py`/`worker.py` (tôi review).
3. Converter/quantize PR của backend phải kèm R6 test; tôi viết template test round-trip.
4. Sampler read-only `/proc/meminfo` mỗi 60s trong 24-48h (không cần GPU lock) để đo dao động MemAvailable do service khác → hiệu chỉnh warn/abort.
5. Thực nghiệm nhỏ (có GPU lock, model tiny): đo `mem_get_info` vs `MemAvailable`, hành vi `set_per_process_memory_fraction`, hiệu quả `expandable_segments` trên GB10 → cập nhật KB.

## 6. Câu hỏi mở
1. bao-backend: trainer (AutoModel) có callback mỗi step để poll `should_stop` + save ckpt không, hay phải wrap? Exit code 75 có xung đột gì không?
3. ~~Ngưỡng~~ — PM đề xuất, fixcode đồng ý: margin `max(10GB, 20%×peak)`, SwapFree ≥2GB, warn 14GB, abort 8GB. Hiệu chỉnh lại sau khi có số liệu dao động `MemAvailable` của host (sampler §5.4).
2. ~~bao-tester: ngưỡng R6~~ — đã cập nhật theo review tester. (Máy chung — tôi nghiêng thà dừng sớm còn hơn treo host.)
4. ~~bao-cleanup: retention R8~~ — đã đồng ý (STATUS file, interrupted status, preflight chung 1 code path).
