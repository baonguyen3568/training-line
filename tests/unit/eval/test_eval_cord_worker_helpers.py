"""Host-safe helpers of scripts/qa/eval_cord_worker.py (the rest runs inside the worker image)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

WORKER = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "eval_cord_worker.py"


def _worker() -> Any:
    spec = importlib.util.spec_from_file_location("eval_cord_worker", WORKER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("ids", "pad", "expected"),
    [
        ([5, 6, 1, 1, 1], 1, ([5, 6], 3)),  # Parse-2.0 pad tail
        ([1, 5, 1], 1, ([1, 5], 1)),  # only the tail, never inner pads
        ([], 1, ([], 0)),
        ([5, 1], None, ([5, 1], 0)),  # tokenizer without a pad token
        ([1, 1], 1, ([], 2)),
    ],
)
def test_strip_trailing_pad(ids: list[int], pad: int | None, expected: tuple[list[int], int]) -> None:
    assert _worker().strip_trailing_pad(ids, pad) == expected


def test_local_dir_refuses_hub_ids(tmp_path: Path) -> None:
    worker = _worker()
    assert worker._local_dir(str(tmp_path), "model") == tmp_path.resolve()
    with pytest.raises(SystemExit, match="local snapshot directory"):
        worker._local_dir("nvidia/NVIDIA-Nemotron-Parse-v1.1", "model")


def test_generation_config_drift_flags_lost_decoding_keys() -> None:
    generation_config_drift = _worker().generation_config_drift
    base = {"repetition_penalty": 1.1, "forced_eos_token_id": 2, "use_cache": True, "transformers_version": "4.51.3"}
    regenerated = {"use_cache": False, "transformers_version": "5.12.1"}  # O-1b export before 23e6aa6
    assert generation_config_drift(regenerated, base) == ["forced_eos_token_id", "repetition_penalty"]
    assert generation_config_drift({**base, "use_cache": False}, base) == []
    # 23e6aa6 exports keep these from the model config; they do not change the generated tokens
    assert generation_config_drift({**base, "output_attentions": False, "output_hidden_states": False}, base) == []
