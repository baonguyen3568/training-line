"""Gate (b) parity for Nemotron-Parse (001a §10.5): the decoder ids that start generation must be identical on
three paths: the training collate, the P-2 eval worker, and NVIDIA's own convention (example.py / test_golden.py:
``tokenizer(task_prompt, add_special_tokens=False)`` with ``decoder_start_token_id`` 2).

Before the collate patch the training prefix was ``[0, 2, 0, ...]`` (processor-added ``<s>``) and HF generate()
prepended 2 at eval: found by the P-2 pipeline check (run 20261012-1809-p2check-500e). CPU only, offline.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nemo_automodel")

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"  # NVIDIA example.py / test_golden.py
EXPECTED = [2, 0, 50004, 50008, 50001]  # decoder_start (</s>) + <s> + three task tokens


def _worker() -> Any:
    spec = importlib.util.spec_from_file_location("eval_cord_worker", REPO / "scripts" / "qa" / "eval_cord_worker.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def processor() -> Any:
    if not PREPARED_V11.is_dir():
        pytest.skip("prepared Nemotron-Parse v1.1 snapshot not available")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)


def _conversation() -> dict[str, Any]:
    from PIL import Image

    target = "<s_menu><s_nm>Latte</s_nm><s_price>35.000</s_price></s_menu>"
    return {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (64, 64), "white")}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
    }["conversation"]


def test_nvidia_golden_prefix(processor: Any) -> None:
    """NVIDIA passes tokenizer(task_prompt) as decoder_input_ids; it already starts with decoder_start (2), so
    generate() prepends nothing."""
    config = json.loads((PREPARED_V11 / "config.json").read_text())
    golden = processor.tokenizer(TASK_PROMPT, add_special_tokens=False).input_ids
    assert golden == EXPECTED
    assert golden[0] == config["decoder_start_token_id"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="collate patch pending (fixcode, 001a §10.5)")
def test_training_and_eval_prefix_equal_nvidia(processor: Any) -> None:
    prefix, _ = _worker().train_decoder_prefix(_conversation(), processor)
    assert prefix.tolist() == EXPECTED, f"training/eval prefix {prefix.tolist()} != NVIDIA {EXPECTED}"
