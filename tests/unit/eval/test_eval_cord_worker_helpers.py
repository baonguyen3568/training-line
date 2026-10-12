"""Host-safe helpers of scripts/qa/eval_cord_worker.py (the rest runs inside the worker image)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
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


class _FakeGenerationConfig:
    """Stands in for transformers.GenerationConfig: from_pretrained reads a JSON dict from the directory."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)

    @classmethod
    def from_pretrained(cls, path: str) -> _FakeGenerationConfig:
        return cls(json.loads((Path(path) / "generation_config.json").read_text()))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def test_repetition_penalty_override_is_recorded_and_does_not_hide_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(GenerationConfig=_FakeGenerationConfig))
    base, ckpt = tmp_path / "base", tmp_path / "ckpt"
    for d, values in ((base, {"repetition_penalty": 1.1, "num_beams": 1}), (ckpt, {"num_beams": 1})):
        d.mkdir()
        (d / "generation_config.json").write_text(json.dumps(values))
    model = types.SimpleNamespace()
    args = types.SimpleNamespace(
        generation_config=str(base),
        max_new_tokens=1024,
        repetition_penalty=1.0,
        require_no_drift=False,
        use_cache="false",
    )
    decoding = _worker()._set_decoding(model, ckpt, args)
    assert model.generation_config.repetition_penalty == 1.0  # the override is what generate() will use
    assert decoding["overrides"] == {"max_new_tokens": 1024, "use_cache": False, "repetition_penalty": 1.0}
    assert model.generation_config.use_cache is False  # pinned from the flag, not inherited from either config
    assert decoding["checkpoint_drift"] == ["repetition_penalty"]  # drift is checkpoint vs base, before overrides
    args.require_no_drift = True
    with pytest.raises(SystemExit, match="repetition_penalty"):
        _worker()._set_decoding(types.SimpleNamespace(), ckpt, args)


@pytest.mark.parametrize("text", ["0", "-1", "nan"])
def test_repetition_penalty_must_be_positive(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _worker()._positive_float(text)
