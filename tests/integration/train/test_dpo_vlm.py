"""DPO for VLMs (ADR-022 #7, vlm_pipeline.train.dpo_vlm): loss terms, answer log-probs, LoRA-off reference, collate.

Runs INSIDE vlm-automodel (offline, CPU; the collate test needs the pinned Qwen3.5-2B processor, no weights).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")

QWEN = Path(__file__).resolve().parents[3] / (
    ".cache/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)


def test_dpo_loss_is_log2_at_the_reference_and_falls_when_chosen_gains() -> None:
    from vlm_pipeline.train.dpo_vlm import dpo_terms

    zero = torch.zeros(3)
    loss, _, wins = dpo_terms(zero, zero, zero, zero, beta=0.1)
    assert torch.allclose(loss, torch.full((3,), math.log(2)))
    assert wins.sum() == 0
    better, _, wins = dpo_terms(torch.ones(3), zero, zero, zero, beta=0.1)
    assert (better < loss).all()
    assert wins.sum() == 3


def test_sequence_logps_sum_only_the_label_tokens() -> None:
    from vlm_pipeline.train.dpo_vlm import sequence_logps

    logits = torch.zeros(1, 3, 4)  # uniform: log p = -log 4 per token
    labels = torch.tensor([[-100, 2, 1]])
    logps, counts = sequence_logps(logits, labels)
    assert counts.tolist() == [2]
    assert torch.allclose(logps, torch.tensor([-2 * math.log(4)]))


def test_chunked_logps_equal_the_cross_entropy_on_the_label_tokens() -> None:
    from vlm_pipeline.train.dpo_vlm import sequence_logps

    torch.manual_seed(0)
    logits = torch.randn(2, 700, 50, dtype=torch.bfloat16)
    labels = torch.randint(0, 50, (2, 700))
    labels[:, :300] = -100  # the prompt
    logps, counts = sequence_logps(logits, labels, chunk=256)
    ce = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    assert torch.allclose(-logps, ce.sum(-1), rtol=1e-4)
    assert counts.tolist() == [400, 400]


def test_checkpointed_chunks_give_the_same_gradient() -> None:
    from vlm_pipeline.train.dpo_vlm import sequence_logps

    torch.manual_seed(0)
    labels = torch.randint(0, 50, (2, 600))
    labels[:, :100] = -100
    base = torch.randn(2, 600, 50)
    grads = []
    for chunk in (256, 10_000):  # chunked + checkpointed vs one piece
        logits = base.clone().requires_grad_(True)
        sequence_logps(logits, labels, chunk=chunk)[0].sum().backward()
        grads.append(logits.grad)
    assert torch.allclose(grads[0], grads[1], rtol=1e-3, atol=1e-6)


def test_reference_is_the_model_with_lora_off_and_scale_comes_back() -> None:
    from nemo_automodel.components._peft.lora import LinearLoRA

    from vlm_pipeline.train.dpo_vlm import lora_disabled

    base = torch.nn.Linear(4, 4, bias=False)
    lora = LinearLoRA(base, dim=2, alpha=4)
    torch.nn.init.ones_(lora.lora_B.weight)  # a non-zero delta
    x = torch.randn(2, 4)
    with lora_disabled(lora):
        assert torch.allclose(lora(x), base(x))
    assert lora.scale == 2.0
    assert not torch.allclose(lora(x), base(x))
    with pytest.raises(RuntimeError, match="needs LoRA"), lora_disabled(torch.nn.Linear(2, 2)):
        pass


def test_collate_puts_chosen_rows_first_with_their_own_labels() -> None:
    from PIL import Image
    from transformers import AutoProcessor

    from vlm_pipeline.train.dpo_vlm import PAIR_KEY, conversation, dpo_collate_fn

    if not (QWEN / "config.json").is_file():
        pytest.skip(f"pinned snapshot not found at {QWEN}")
    processor = AutoProcessor.from_pretrained(str(QWEN), min_pixels=200704, max_pixels=1003520)
    image = Image.new("RGB", (640, 480), (255, 255, 255))
    rows = [{"chosen": conversation(image, "Total?", "45,500"), "rejected": conversation(image, "Total?", "54,500")}]
    batch = dpo_collate_fn(rows, processor)
    assert batch[PAIR_KEY] == 1
    assert batch["input_ids"].shape[0] == 2
    decoded = [processor.tokenizer.decode(r[r != -100]) for r in batch["labels"]]
    assert "45,500" in decoded[0]
    assert "54,500" in decoded[1]
