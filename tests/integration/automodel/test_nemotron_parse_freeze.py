"""Regression tests for third_party/patches/automodel/0002 (runs INSIDE vlm-automodel, offline, CPU/meta only).

P-1 incident (2026-10-12, 001a §9): `freeze_vision_tower` matched nothing in Nemotron-Parse (backbone is
`encoder.model_encoder`), so 100% of 936M params trained and 32 un-checkpointed RADIO blocks kept activations
(+~20 GB). Unknown `freeze_config` keys (`freeze_embeddings`) were ignored silently.
Must FAIL on 26.08-2 and PASS on 26.08-3.
Expected shares (bao-research, plan A: freeze RADIO backbone only, neck + embeddings train): v1.1 = 284.6M / 936.3M.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "model_configs" / "nemotron_parse"
RADIO_DIR = Path(
    os.environ.get(
        "VLM_CRADIO_V2H_DIR",
        str(REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "c-radiov2-h_0d8f4c18"),
    )
)
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"


def _build(version: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    import torch
    from nemo_automodel.components.models.nemotron_parse.model import (
        NemotronParseConfig,
        NemotronParseForConditionalGeneration,
    )

    if not (RADIO_DIR / "config.json").is_file():
        pytest.skip(f"pinned C-RADIOv2-H copy not found at {RADIO_DIR}")
    cfg = json.loads((FIXTURES / f"config_{version}.json").read_text())
    cfg["encoder"]["_name_or_path"] = str(RADIO_DIR)
    cfg["encoder"]["auto_map"] = {
        "AutoConfig": f"{RADIO_DIR}--hf_model.RADIOConfig",
        "AutoModel": f"{RADIO_DIR}--hf_model.RADIOModel",
    }
    d = tmp_path / version
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    with torch.device("meta"):
        return NemotronParseForConditionalGeneration(NemotronParseConfig.from_pretrained(d, trust_remote_code=True))


def _pct(model) -> float:  # type: ignore[no-untyped-def]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 100.0 * trainable / total


@pytest.mark.parametrize(("version", "expected_pct"), [("v1.1", 30.4), ("v2.0", None)])
def test_freeze_vision_tower_freezes_only_the_radio_backbone(
    version: str, expected_pct: float | None, tmp_path: Path
) -> None:
    from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

    model = _build(version, tmp_path)
    apply_parameter_freezing(model, {"freeze_vision_tower": True})
    for name, p in model.named_parameters():
        backbone = name.startswith("encoder.model_encoder.")
        assert p.requires_grad is not backbone, name  # backbone frozen; neck, decoder, embeddings, head train
    if expected_pct is not None:
        assert _pct(model) == pytest.approx(expected_pct, abs=1.0)
    else:  # 2.0 (tied head): reported for the recipe catalog, must be well below 100 %
        assert 20.0 < _pct(model) < 50.0


def test_unknown_freeze_key_is_an_error(tmp_path: Path) -> None:
    from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

    model = _build("v1.1", tmp_path)
    with pytest.raises(ValueError, match="freeze_embeddings"):
        apply_parameter_freezing(model, {"freeze_vision_tower": True, "freeze_embeddings": True})


def test_trainable_share_outside_expectation_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from nemo_automodel.components.utils.model_utils import print_trainable_parameters

    model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 10))  # 220 params, all trainable
    monkeypatch.setenv("VLM_EXPECTED_TRAINABLE_PCT", "29.4:31.4")
    with pytest.raises(RuntimeError, match="VLM_TRAINABLE_PCT_MISMATCH"):
        print_trainable_parameters(model)  # 100 % trainable, like the P-1 incident
    model[0].requires_grad_(False)
    monkeypatch.setenv("VLM_EXPECTED_TRAINABLE_PCT", "49:51")
    print_trainable_parameters(model)  # 50 %: inside the range, no error


@pytest.mark.parametrize("value", ["30", "30:abc", "31.4:29.4", "-1:30", "30:101"])
def test_malformed_trainable_share_expectation_is_an_error(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the runner's env must not disable the guard or surface as an unrelated unpacking error."""
    import torch
    from nemo_automodel.components.utils.model_utils import print_trainable_parameters

    monkeypatch.setenv("VLM_EXPECTED_TRAINABLE_PCT", value)
    with pytest.raises(ValueError, match="VLM_EXPECTED_TRAINABLE_PCT"):
        print_trainable_parameters(torch.nn.Linear(10, 10))


def test_label_mask_keeps_exactly_the_target_plus_eos() -> None:
    """The collate warns "Falling back to BPE pattern-match labels" but masks correctly; lock that behaviour."""
    if not PREPARED_V11.is_dir():
        pytest.skip("prepared Nemotron-Parse v1.1 dir not found")
    from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_parse_collate_fn
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)
    tok = processor.tokenizer
    targets = ["<s_menu><s_nm>Cafe Latte</s_nm><s_price>4.50</s_price></s_menu>", "<s_menu><s_nm>Tea</s_nm></s_menu>"]
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480), "white")}]},
                {"role": "assistant", "content": [{"type": "text", "text": t}]},
            ]
        }
        for t in targets
    ]
    batch = nemotron_parse_collate_fn(examples, processor)
    for i, target in enumerate(targets):
        labels = batch["labels"][i]
        kept = labels[labels != -100].tolist()
        assert tok.decode(kept) == target + tok.eos_token  # prompt + padding masked, EOS learned
