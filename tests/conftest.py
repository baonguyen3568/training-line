"""Shared pytest setup (ADR-004 A1/A2, S-8).

- ``gpu`` tests are skipped when CUDA is unavailable (host venv has no torch) and, when CUDA is present,
  fail fast unless the GPU lock is held by the parent host process: the host writes a random token to
  ``.team/GPU_LOCK`` and passes it as ``VLM_GPU_LOCK_TOKEN``; tests never acquire the lock themselves
  (PID namespaces differ inside the container).
- Every test starts from a fixed seed.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

SEED = 1234
GPU_LOCK_ENV_VAR = "VLM_GPU_LOCK_TOKEN"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs CUDA: run via scripts/check.sh --container")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    random.seed(SEED)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(SEED)
    if "torch" in sys.modules:  # seed torch only if a test already imported it; never import it just to seed
        sys.modules["torch"].manual_seed(SEED)


@pytest.fixture(autouse=True)
def _gpu_lock_held(request: pytest.FixtureRequest) -> None:
    """For ``gpu`` tests only: verify the lock token via common.gpu_lock (never parse the file here)."""
    if request.node.get_closest_marker("gpu") is None:
        return
    token = os.environ.get(GPU_LOCK_ENV_VAR)
    if not token:
        pytest.fail(f"{GPU_LOCK_ENV_VAR} not set: gpu tests must be launched by the host holding .team/GPU_LOCK")
    from vlm_pipeline.common.gpu_lock import GpuLockError, default_lock_path, verify_token

    try:
        verify_token(default_lock_path(), token)
    except GpuLockError as exc:
        pytest.fail(f"GPU lock not held for this run ({exc.code}): {exc}")
