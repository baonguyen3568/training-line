"""bench.perf container-side statistics (host-safe parts of serve/bench.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.serve.bench import MemSampler, mem_available_gb, percentile, summarize


def test_nearest_rank_percentile_returns_observed_values() -> None:
    values = [float(v) for v in range(1, 51)]  # 1..50
    assert percentile(values, 50) == 25.0
    assert percentile(values, 90) == 45.0
    assert percentile(values, 99) == 50.0
    assert percentile([7.0], 99) == 7.0
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


def test_summary_metrics() -> None:
    timings = [
        {"prepare_s": 0.1, "generate_s": 0.9, "e2e_s": 1.0, "new_tokens": 90.0},
        {"prepare_s": 0.1, "generate_s": 1.9, "e2e_s": 2.0, "new_tokens": 190.0},
    ]
    s = summarize(timings)
    assert s["e2e_ms_p50"] == 1000.0
    assert s["e2e_ms_max"] == 2000.0
    assert s["tokens_per_s"] == 100.0  # 280 tokens / 2.8 s of generate
    assert s["requests_per_s"] == pytest.approx(0.667, abs=1e-3)
    assert s["output_tokens_mean"] == 140.0


def _meminfo(path: Path, avail_gb: float) -> Path:
    path.write_text(f"MemTotal: {128 * 1024 * 1024} kB\nMemAvailable: {int(avail_gb * 1024 * 1024)} kB\n")
    return path


def test_mem_sampler_keeps_the_minimum(tmp_path: Path) -> None:
    path = _meminfo(tmp_path / "meminfo", 50.0)
    assert mem_available_gb(path) == pytest.approx(50.0)
    sampler = MemSampler(path=path, interval_s=3600)  # no background ticks during the test
    with sampler:
        _meminfo(path, 30.0)
        sampler.sample()
        _meminfo(path, 45.0)
    assert sampler.min_gb == pytest.approx(30.0)
