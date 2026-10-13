"""Speed / memory benchmark (node ``bench.perf``, B-6) — runs INSIDE the training image, one shot, under the GPU lock.

``python -m vlm_pipeline.serve.bench <request.json>`` writes ``result.json`` next to the request. It times exactly the
code path of "Chạy thử" (``serve.trial.load_parse / prepare_parse / generate_parse``), so the numbers describe what
the user runs (tester-perf-bench protocol):
- load once (timed), ``warmup`` requests discarded (CUDA context, kernel autotune / JIT), then ``num_requests`` timed
  one at a time (batch 1) with ``torch.cuda.synchronize()`` around each phase: prepare (processor on CPU + copy to the
  GPU), generate, end to end;
- memory on UMA: host MemAvailable sampled every 0.5 s (baseline before load, after load, minimum while timing), plus
  ``torch.cuda.max_memory_allocated`` (allocator view); nvidia-smi reports no memory on GB10;
- images: the first ``len(image_ids)`` CORD-v2 test images (fixed, so runs compare), cycled.
Only Nemotron-Parse (engine hf) for now; chat models go through vLLM once the server exists (B7-3).
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vlm_pipeline.serve.trial import TrialRequest, generate_parse, load_parse, prepare_parse

MEMINFO = Path("/proc/meminfo")
SAMPLE_INTERVAL_S = 0.5
BENCH_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BenchRequest:
    trial: dict[str, Any]  # TrialRequest fields (model, base processor + generation config, template overrides)
    data_dir: str  # pinned CORD-v2 snapshot (container path)
    image_ids: list[int]
    num_requests: int
    warmup: int = 3
    split: str = "test"

    @classmethod
    def from_json(cls, text: str) -> BenchRequest:
        return cls(**json.loads(text))


def mem_available_gb(path: Path = MEMINFO) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise ValueError("MemAvailable missing in /proc/meminfo")


@dataclass
class MemSampler:
    """Background minimum of host MemAvailable (UMA: includes what CUDA allocates)."""

    path: Path = MEMINFO
    interval_s: float = SAMPLE_INTERVAL_S
    min_gb: float = float("inf")
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def sample(self) -> float:
        value = mem_available_gb(self.path)
        self.min_gb = min(self.min_gb, value)
        return value

    def __enter__(self) -> MemSampler:
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                self.sample()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0-100) — no interpolation, so p99 of 50 samples is an observed value."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil(n*q/100), at least 1
    return ordered[int(rank) - 1]


def summarize(timings: list[dict[str, float]]) -> dict[str, float]:
    """Flat perf metrics (frontend-metrics-contract names) from per-request timings in seconds."""
    out: dict[str, float] = {}
    for key, name in (("e2e_s", "e2e_ms"), ("prepare_s", "prepare_ms"), ("generate_s", "generate_ms")):
        ms = [1000 * t[key] for t in timings]
        for q in (50, 90, 99):
            out[f"{name}_p{q}"] = round(percentile(ms, q), 1)
        out[f"{name}_min"] = round(min(ms), 1)
        out[f"{name}_max"] = round(max(ms), 1)
    tokens = sum(t["new_tokens"] for t in timings)
    out["tokens_per_s"] = round(tokens / sum(t["generate_s"] for t in timings), 2)
    out["requests_per_s"] = round(len(timings) / sum(t["e2e_s"] for t in timings), 3)
    out["output_tokens_mean"] = round(statistics.fmean(t["new_tokens"] for t in timings), 1)
    return out


def run(request_path: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset

    req = BenchRequest.from_json(request_path.read_text(encoding="utf-8"))
    trial = TrialRequest(**req.trial)
    ds = load_dataset(req.data_dir, split=req.split)
    images = [ds[i]["image"].convert("RGB") for i in req.image_ids]

    def one(image: Any) -> dict[str, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = prepare_parse(session, image)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        new = generate_parse(session, inputs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {"prepare_s": t1 - t0, "generate_s": t2 - t1, "e2e_s": t2 - t0, "new_tokens": float(len(new))}

    with MemSampler() as mem:
        baseline_gb = mem.sample()
        t_load = time.perf_counter()
        session = load_parse(trial)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load
        after_load_gb = mem.sample()
        for i in range(req.warmup):
            one(images[i % len(images)])
        startup_min_gb = mem.min_gb
        steady_gb = mem.sample()
        torch.cuda.reset_peak_memory_stats()
        mem.min_gb = steady_gb
        timings = [one(images[i % len(images)]) for i in range(req.num_requests)]
    result = {
        **summarize(timings),
        "load_s": round(load_s, 2),
        "torch_max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "host_baseline_available_gb": round(baseline_gb, 2),
        "memory_startup_peak_gb": round(baseline_gb - startup_min_gb, 2),  # load + warmup (JIT, autotune)
        "memory_steady_gb": round(baseline_gb - after_load_gb, 2),  # weights + context after load
        "memory_peak_load_gb": round(baseline_gb - mem.min_gb, 2),  # while timing
        "per_request": timings,
        "decoding": {
            k: getattr(session.config, k, None)
            for k in ("do_sample", "num_beams", "repetition_penalty", "max_new_tokens", "use_cache")
        },
        "overrides": session.overrides,
        "model_class": f"{type(session.model).__module__}.{type(session.model).__qualname__}",
        "conditions": {
            "engine": "hf",
            "batch_size": 1,
            "warmup": req.warmup,
            "num_requests": req.num_requests,
            "images": f"cord-v2:{req.split}[{len(images)}]",
            "precision": "bf16",
        },
        "request": asdict(req),
        "bench_sha256": BENCH_SHA256,
    }
    (request_path.parent / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m vlm_pipeline.serve.bench <request.json>", file=sys.stderr)
        return 2
    from vlm_pipeline.common.cuda_init import ensure_cuda_context

    ensure_cuda_context()  # before the checkpoint is loaded on the CPU: GB10 context creation (KB fixcode §2.2b)
    run(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
