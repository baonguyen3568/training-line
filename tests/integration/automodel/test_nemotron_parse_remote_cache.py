"""Nemotron-Parse v1.1 through the HF-remote class (the CORD eval path): KV-cached generation must equal uncached.

The remote decoder (hf_nemotron_parse_modeling.py, NVIDIA's port to transformers 5.x with EncoderDecoderCache) has no
positional embedding, so cached and uncached greedy decoding are the same computation. The cached path had never been
checked against anything (the parity gate is teacher-forced), while eval_cord_worker defaults to use_cache=true.
Inputs follow NVIDIA's test_golden.py: its deterministic image, `generate(**processor(...))` (the processor's
attention_mask is forwarded as decoder_attention_mask by the remote prepare_inputs_for_generation), the model's own
generation config, 50 new tokens. Also the eval worker's call (pixel_values + decoder_input_ids, no mask).
GPU bfloat16 (golden was captured on cuda bf16): cache and no-cache, both calls, must reproduce NVIDIA's ids.
A CPU float32 cache == no-cache variant passed once (26.08-5, 2026-10-13, both calls) but took 2h36m (RADIO-H on
CPU), too slow for a gate, so it is not kept.
Coverage: the golden is 23 new tokens (EOS before 50), so this checks the cache on a short sequence only; long-sequence
cache equivalence is not covered here (evidence for that: the P-3 s299 cache vs no-cache A/B, bao-tester 2026-10-12).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
MAX_NEW_TOKENS = 50  # test_golden.py MAX_NEW_TOKENS_GOLDEN


def _golden() -> dict[str, Any]:
    return dict(json.loads((PREPARED_V11 / "golden_outputs.json").read_text())["generation"])


def _golden_image() -> Any:
    spec = importlib.util.spec_from_file_location("nvidia_test_golden", PREPARED_V11 / "test_golden.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_test_image()


def _load(device: str, dtype_name: str) -> tuple[Any, Any]:
    if not (PREPARED_V11 / "golden_outputs.json").is_file():
        pytest.skip("prepared Nemotron-Parse v1.1 snapshot with NVIDIA's golden_outputs.json not found")
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)
    model = AutoModel.from_pretrained(PREPARED_V11, trust_remote_code=True, dtype=getattr(torch, dtype_name))
    model = model.to(device).eval()
    assert type(model).__module__.endswith("hf_nemotron_parse_modeling"), type(model)  # remote, not native
    return model, processor


def _generate(model: Any, processor: Any, *, use_cache: bool, worker_call: bool) -> list[int]:
    import torch

    inputs = processor(images=[_golden_image()], text=TASK_PROMPT, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if worker_call:  # eval_cord_worker: no attention mask, explicit decoder prefix
        inputs = {"pixel_values": inputs["pixel_values"], "decoder_input_ids": inputs["input_ids"]}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, use_cache=use_cache)
    return [int(t) for t in out[0].tolist()]


@pytest.fixture(scope="module")
def gpu_model() -> tuple[Any, Any]:
    return _load("cuda", "bfloat16")


@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
@pytest.mark.parametrize("worker_call", [False, True], ids=["nvidia_call", "worker_call"])
def test_generation_matches_nvidia_golden_on_gpu(
    gpu_model: tuple[Any, Any], use_cache: bool, worker_call: bool
) -> None:
    model, processor = gpu_model
    ids = _generate(model, processor, use_cache=use_cache, worker_call=worker_call)
    assert ids[:5] == [2, 0, 50004, 50008, 50001]  # NVIDIA's prefix kept, no extra start token
    assert ids == _golden()["token_ids"]
