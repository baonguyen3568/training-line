"""GB10 workaround: a GPU job creates its CUDA context before loading anything large on the CPU (KB fixcode §2.2b)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from vlm_pipeline.common.cuda_init import ensure_cuda_context


def _fake_torch(monkeypatch: pytest.MonkeyPatch, *, available: bool, failures: int) -> list[str]:
    calls: list[str] = []
    left = {"n": failures}

    def empty(*a: Any, **kw: Any) -> None:
        calls.append(f"empty:{kw.get('device')}")
        if left["n"] > 0:
            left["n"] -= 1
            raise RuntimeError("CUDA error: out of memory\nmore")

    cuda = SimpleNamespace(is_available=lambda: available, synchronize=lambda: calls.append("sync"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, empty=empty))
    monkeypatch.setattr("time.sleep", lambda s: calls.append(f"sleep:{s}"))
    return calls


def test_the_context_is_created_with_a_tiny_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_torch(monkeypatch, available=True, failures=0)
    assert ensure_cuda_context()
    assert calls == ["empty:cuda", "sync"]


def test_no_gpu_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_torch(monkeypatch, available=False, failures=0)
    assert not ensure_cuda_context()
    assert calls == []


def test_a_failed_creation_is_retried_in_process_then_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_torch(monkeypatch, available=True, failures=1)
    assert ensure_cuda_context(attempts=3, pause_s=5.0)
    assert calls == ["empty:cuda", "sleep:5.0", "empty:cuda", "sync"]
    _fake_torch(monkeypatch, available=True, failures=5)
    with pytest.raises(RuntimeError, match="out of memory"):
        ensure_cuda_context(attempts=2, pause_s=0.0)


def test_trial_and_bench_create_the_context_before_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from vlm_pipeline.serve import bench, trial

    order: list[str] = []
    monkeypatch.setattr("vlm_pipeline.common.cuda_init.ensure_cuda_context", lambda: order.append("ctx") or True)
    monkeypatch.setattr(trial, "run", lambda p: order.append("trial") or {})
    monkeypatch.setattr(bench, "run", lambda p: order.append("bench"))
    assert trial.main([str(tmp_path / "r.json")]) == 0
    assert bench.main([str(tmp_path / "r.json")]) == 0
    assert order == ["ctx", "trial", "ctx", "bench"]
