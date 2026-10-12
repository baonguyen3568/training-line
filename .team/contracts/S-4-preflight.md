# S-4 preflight contract — SOURCE OF TRUTH (bao-fixcode + bao-cleanup, 2026-10-12)

Messages crossed several times; **this file wins over any chat message**. Change it only by editing here + notifying the other owner.

## Ownership
| File | Owner | Reviewer |
|------|-------|----------|
| `src/vlm_pipeline/common/errors.py` (`ExitCode`, `VlmPipelineError`, `InsufficientMemoryError`) | bao-fixcode | bao-cleanup |
| `src/vlm_pipeline/common/memguard.py` (incl. `preflight`) | bao-fixcode | bao-cleanup |
| `src/vlm_pipeline/common/diskguard.py` | bao-cleanup | bao-fixcode |
| `scripts/cleanup/preflight.py`, `scripts/cleanup/disk_report.py` | bao-cleanup | bao-fixcode |
| `src/vlm_pipeline/common/gpu_lock.py` (S-2) | bao-backend | — |

Import direction: `memguard → diskguard` (and later `→ gpu_lock`). `diskguard` imports nothing from `memguard`. Stdlib only.

## Entry point (the only one callers use)
```python
memguard.preflight(est_peak_gb: float, *, ckpt_gb: float = 0.0,
                   snapshot: MemSnapshot | None = None,
                   disk: DiskSnapshot | None = None,
                   lock_holder: Callable[[], LockStatus | None] | None = None,
                   root: Path | None = None,            # repo root; tests pass tmp_path
                   thresholds: MemThresholds = DEFAULT_THRESHOLDS) -> Verdict
```
- `disk=None` → `diskguard.read_disk(<project root>)`; `lock_holder=None` → `diskguard.check_lock(.team/GPU_LOCK)` (later: `gpu_lock`).
- Exit-code priority when several fail: `LOCK_BUSY(4) > NO_MEM(2) > NO_DISK(3)`; `Verdict.reasons` lists **all** failures.
- `Verdict = {ok, exit_code, reasons: list[str], findings, mem, disk, lock, thresholds}` + `to_json()`.
- Units: every `*_gb` value is **GiB** (1024³) in both memguard and diskguard.
- Memory thresholds (ADR-005): margin = max(10 GB, 20% × est_peak), hard floor MemAvailable ≥ 20 GB, SwapFree ≥ 2 GB.

## diskguard API (bao-cleanup)
```python
@dataclass(frozen=True)
class DiskSnapshot: path: str; total_gb: float; free_gb: float

def read_disk(path: Path, *, statvfs: Callable[[str], os.statvfs_result] = os.statvfs) -> DiskSnapshot
def required_disk_gb(ckpt_gb: float, *, max_recent: int = 2, headroom_gb: float = 50.0) -> float
    # = ckpt_gb * (max_recent + 1) + headroom_gb ; ckpt_gb=0 still needs headroom
def check_disk(required_gb: float, snapshot: DiskSnapshot) -> list[str]   # reasons; empty = OK

@dataclass(frozen=True)
class LockStatus: who: str; job: str; run_id: str | None; host_pid: int | None; started_at: str | None; alive: bool; corrupt: bool = False

def check_lock(lock_path: Path, *, pid_alive: Callable[[int], bool] = _pid_alive) -> LockStatus | None
    # read-only; accepts JSON lock or legacy text "who | job | started_at | PID"; None = no lock; dead PID → alive=False (still LOCK_BUSY, "stale lock — tell PM", NEVER delete);
    # unreadable/corrupt → corrupt=True, alive=True (treated busy). Temporary until gpu_lock.read_lock/is_alive (S-2).
```

## Wrappers (bao-cleanup) — implemented 2026-10-12
- `scripts/cleanup/preflight.py --est-peak-gb N --ckpt-gb M` → prints `Verdict.to_json()`, exits with `ExitCode` (default est 40 GB, ADR-005 D8).
- `scripts/cleanup/disk_report.py [--no-write]` → `outputs/system/disk_report.json` (schema v1, `common/storage_report.py`); docker images deduped by shared base layers.
