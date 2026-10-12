"""Gate (d): Nemotron-Parse label mask on REAL CORD-v2 samples (runs INSIDE vlm-automodel, offline, CPU).

AutoModel's collate cannot derive chat-template turn markers for the Parse processor and falls back to BPE
pattern-matching (warning "could not derive turn markers ... may produce nan loss"); the code itself calls that path
fragile. Verified correct on 26.08-3 (2026-10-12, bao-fixcode): labels decode to exactly the json2token target +
EOS, without task/prompt tokens. This locks it so a bump of AutoModel or the tokenizer cannot break it silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
CORD = (
    REPO / ".cache" / "hf" / "hub" / "datasets--naver-clova-ix--cord-v2" / "snapshots"
    / "7f0115a4b758a71d6473b8d085751692da2fef98"
)  # fmt: skip
PROMPT_TOKENS = ("<predict_bbox>", "<predict_classes>", "<output_markdown>", "<output_plain>")


@pytest.fixture(scope="module")
def processor() -> Any:
    if not PREPARED_V11.is_dir():
        pytest.skip("prepared Nemotron-Parse v1.1 dir not found")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)


def _assistant_text(example: dict[str, Any]) -> str:
    msg = next(m for m in example["conversation"] if m["role"] == "assistant")
    content = msg["content"]
    return str(content[0]["text"] if isinstance(content, list) else content)


def _check(examples: list[dict[str, Any]], processor: Any) -> list[str]:
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn

    tok = processor.tokenizer
    batch = nemotron_parse_collate_fn(examples, processor)
    targets = []
    for i, example in enumerate(examples):
        target = _assistant_text(example)
        labels = batch["labels"][i]
        decoded = tok.decode(labels[labels != -100].tolist())
        assert decoded == target + tok.eos_token, (i, decoded[:80], target[:80])
        assert not any(t in decoded for t in PROMPT_TOKENS), i
        targets.append(target)
    return targets


def test_real_cord_batch_labels_are_exactly_target_plus_eos(processor: Any) -> None:
    if not CORD.is_dir():
        pytest.skip("CORD-v2 @7f0115a4 not in the project HF cache")
    from nemo_automodel.components.datasets.vlm.datasets import make_cord_v2_dataset

    dataset = make_cord_v2_dataset(path_or_dataset=str(CORD), split="train")
    _check([dataset[i] for i in range(4)], processor)


@pytest.mark.parametrize(
    "target",
    [
        "<s_menu><s_nm>Cà phê sữa đá</s_nm><s_price>35.000₫</s_price></s_menu>",
        "<s_menu><s_nm>Bánh mì ốp la</s_nm><s_price>€4,50 ¥1,200 Rp 12.500</s_price></s_menu>",
        "Trà</s_nm>",  # target that does not start with a tag: odd BPE boundary
    ],
)
def test_non_ascii_and_currency_targets(target: str, processor: Any) -> None:
    """Where BPE pattern-matching is most likely to misalign (Vietnamese diacritics, currency signs)."""
    from PIL import Image

    assert any(ord(c) > 127 for c in target) or not target.startswith("<")
    example = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (640, 480), "white")},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
    }
    _check([example], processor)
