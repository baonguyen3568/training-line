"""Patch 0008 (AutoModel VLM collate): assistant-turn labels for families outside Qwen.

Bug (bao-research 2026-10-14, ADR-021 family checks on CORD v2): ``build_labels_from_template`` derived the turn
markers from the TOKENIZER's chat template while the batch is rendered with the PROCESSOR's. LLaVA-OneVision's
processor renders ``<|im_start|>assistant \\n`` (a space) and SmolVLM's tokenizer template renders nothing for string
content, so every label was -100 and training learned nothing, silently.

Runs INSIDE vlm-automodel (offline, CPU; processors only, no weights). Must FAIL on 26.08-7 for SmolVLM and LLaVA-OV
and PASS with patch 0008; Qwen (fast path) and Gemma 4 / InternVL (already fine) must keep the same labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")

HUB = Path(__file__).resolve().parents[3] / ".cache/hf/hub"
ANSWER = "<s_total><s_total_price>45,500</s_total_price></s_total>"
CASES = {
    "smolvlm": ("models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964",
                "default_collate_fn", {}),
    "llava_onevision": ("models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf/snapshots/"
                        "74dd0bf867a4cda7950c17663794267c60cf4b40", "llava_onevision_collate_fn", {}),
    "internvl": ("models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171",
                 "default_collate_fn", {}),
    "gemma4": ("models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7",
               "default_collate_fn", {}),
    "qwen3_5": ("models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc",
                "default_collate_fn", {"min_pixels": 200704, "max_pixels": 1003520}),
}  # fmt: skip


def _example() -> dict:
    from PIL import Image

    image = Image.new("RGB", (640, 480), (255, 255, 255))
    return {"conversation": [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Read the receipt."}]},
        {"role": "assistant", "content": [{"type": "text", "text": ANSWER}]},
    ]}  # fmt: skip


@pytest.mark.parametrize("family", sorted(CASES))
def test_labels_are_the_assistant_answer(family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_automodel.components.datasets.vlm import collate_fns as cf
    from transformers import AutoProcessor

    path, collate, kwargs = CASES[family]
    if not (HUB / path / "config.json").is_file():
        pytest.skip(f"pinned snapshot not found: {path}")
    monkeypatch.setattr(cf, "_LABEL_STATS", {"batches": 0, "empty": 0}, raising=False)
    processor = AutoProcessor.from_pretrained(str(HUB / path), **kwargs)
    batch = getattr(cf, collate)([_example(), _example()], processor)
    for row in batch["labels"]:
        learned = processor.tokenizer.decode(row[row != -100])
        assert ANSWER in learned, f"{family}: labels {learned!r}"


def test_first_batch_without_labels_stops_later_ones_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from nemo_automodel.components.datasets.vlm import collate_fns as cf

    monkeypatch.setattr(cf, "_LABEL_STATS", {"batches": 0, "empty": 0}, raising=False)
    empty, full = torch.full((1, 4), -100), torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="no assistant tokens"):
        cf._require_labels(empty, "P")
    monkeypatch.setattr(cf, "_LABEL_STATS", {"batches": 0, "empty": 0}, raising=False)
    cf._require_labels(full, "P")
    for _ in range(19):
        cf._require_labels(full, "P")
    assert cf._require_labels(empty, "P") is empty  # 1 of 21: warned, kept
    with pytest.raises(ValueError, match="2 of 22"):
        cf._require_labels(empty, "P")  # 2 of 22 > 5 %


def test_broken_template_stops_at_the_first_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every path (processor markers, tokenizer markers, BPE fallback) finds no label: the first batch must raise,
    and the batch is counted once, not once per path (fixcode review of 0008)."""
    import torch
    from nemo_automodel.components.datasets.vlm import collate_fns as cf

    class Tok:
        unk_token_id = 0

        def apply_chat_template(self, *a, **k):  # type: ignore[no-untyped-def]
            return [1, 2, 3]

    class Proc:
        chat_template = "{{ messages }}"
        tokenizer = Tok()

    stats = {"batches": 0, "empty": 0}
    monkeypatch.setattr(cf, "_LABEL_STATS", stats, raising=False)
    monkeypatch.setattr(cf, "_derive_turn_markers_from_processor", lambda p: ([7, 7], 8), raising=False)
    monkeypatch.setattr(cf, "_derive_turn_markers", lambda t: ([7, 7], 8))
    monkeypatch.setattr(cf, "build_labels", lambda ids, conv, p: torch.full_like(ids, -100))
    with pytest.raises(ValueError, match="no assistant tokens"):
        cf.build_labels_from_template(torch.ones((1, 6), dtype=torch.long), [[]], Proc())
    assert stats == {"batches": 1, "empty": 1}
