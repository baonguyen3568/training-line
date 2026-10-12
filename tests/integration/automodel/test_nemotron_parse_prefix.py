"""Patch 0004: Nemotron-Parse training uses NVIDIA's decoder prefix (runs INSIDE vlm-automodel, offline, CPU).

NVIDIA's reference (README, test_golden.py; decoder_start_token_id = 2) encodes the task prompt with
add_special_tokens=False, so the decoder input starts `</s><s><predict_bbox><predict_classes><output_markdown>`
= [2, 0, 50004, 50008, 50001]. AutoModel's collate let the tokenizer add its BOS: [0, 2, 0, 50004, ...] at training
time only (found by bao-tester; decision 001a §10.5: fix training, not serving). Must FAIL on 26.08-3, PASS patched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
NVIDIA_PREFIX = [2, 0, 50004, 50008, 50001]


@pytest.fixture(scope="module")
def processor() -> Any:
    if not PREPARED_V11.is_dir():
        pytest.skip("prepared Nemotron-Parse v1.1 dir not found")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)


def _examples(targets: list[str]) -> list[dict[str, Any]]:
    from PIL import Image

    return [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480), "white")}]},
                {"role": "assistant", "content": [{"type": "text", "text": t}]},
            ]
        }
        for t in targets
    ]


def test_expected_prefix_is_the_prompt_without_special_tokens(processor: Any) -> None:
    assert processor.tokenizer(TASK_PROMPT, add_special_tokens=False)["input_ids"] == NVIDIA_PREFIX


def test_decoder_input_starts_with_the_nvidia_prefix(processor: Any) -> None:
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn

    targets = [
        "<s_menu><s_nm>Tea</s_nm></s_menu>",
        "<s_menu><s_nm>Cà phê sữa đá</s_nm><s_price>35.000₫</s_price></s_menu>",
    ]
    batch = nemotron_parse_collate_fn(_examples(targets), processor)
    for row in range(len(targets)):
        assert batch["decoder_input_ids"][row, : len(NVIDIA_PREFIX)].tolist() == NVIDIA_PREFIX


def test_labels_are_target_plus_eos_and_padding_masked(processor: Any) -> None:
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn

    tok = processor.tokenizer
    targets = ["<s_menu><s_nm>Tea</s_nm></s_menu>", "<s_menu><s_nm>Cafe Latte</s_nm><s_price>4.50</s_price></s_menu>"]
    batch = nemotron_parse_collate_fn(_examples(targets), processor)
    for row, target in enumerate(targets):
        labels = batch["labels"][row]
        assert tok.decode(labels[labels != -100].tolist()) == target + tok.eos_token
        # label[i] is the next token after decoder_input[i]: the first learned token follows the full prompt
        first = int((labels != -100).nonzero()[0])
        assert first == len(NVIDIA_PREFIX) - 1
    assert batch["labels"].shape == batch["decoder_input_ids"].shape


def test_prompt_that_does_not_tokenize_in_place_is_refused(processor: Any) -> None:
    """The label mask is positional: if the prompt ids are not at the start of the sample, refuse to train."""
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn

    class _ShiftedPrompt:  # the tokenizer encodes the prompt alone differently than inside the sample
        def __init__(self, tok: Any) -> None:
            self._tok = tok

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tok, name)

        def __call__(self, text: Any, **kwargs: Any) -> Any:
            out = self._tok(text, **kwargs)
            if text == TASK_PROMPT:
                out["input_ids"] = [*out["input_ids"][:-1], 50002]
            return out

    class _Processor:
        def __init__(self, proc: Any) -> None:
            self._proc = proc
            self.tokenizer = _ShiftedPrompt(proc.tokenizer)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._proc, name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._proc(*args, **kwargs)

    with pytest.raises(ValueError, match="task prompt ids"):
        nemotron_parse_collate_fn(_examples(["<s_menu>Tea</s_menu>"]), _Processor(processor))
