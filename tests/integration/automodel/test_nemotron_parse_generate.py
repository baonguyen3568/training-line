"""BUG-001 regression (patch 0003): Nemotron-Parse native generation must not rely on a KV cache.

NemotronParseDecoder ignores the cache (every layer gets past_key_values=None, use_cache=False), so the default
cached generation fed only the newest token at position 0 and looped (`<x_0.3906>` forever, golden gate (b) on
26.08-2). The patch always feeds the whole decoder sequence. Runs INSIDE vlm-automodel (offline, meta device);
the end-to-end proof is the golden gate on GPU (backend/tester harness). Must FAIL on 26.08-3, PASS on 26.08-4.
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
PREPARED_V11 = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
RADIO_DIR = Path(
    os.environ.get(
        "VLM_CRADIO_V2H_DIR",
        str(REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "c-radiov2-h_0d8f4c18"),
    )
)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    import torch
    from nemo_automodel.components.models.nemotron_parse.model import (
        NemotronParseConfig,
        NemotronParseForConditionalGeneration,
    )

    if not (RADIO_DIR / "config.json").is_file():
        pytest.skip(f"pinned C-RADIOv2-H copy not found at {RADIO_DIR}")
    cfg = json.loads((FIXTURES / "config_v1.1.json").read_text())
    cfg["encoder"]["_name_or_path"] = str(RADIO_DIR)
    cfg["encoder"]["auto_map"] = {
        "AutoConfig": f"{RADIO_DIR}--hf_model.RADIOConfig",
        "AutoModel": f"{RADIO_DIR}--hf_model.RADIOModel",
    }
    d = tmp_path_factory.mktemp("parse")
    (d / "config.json").write_text(json.dumps(cfg))
    with torch.device("meta"):
        return NemotronParseForConditionalGeneration(NemotronParseConfig.from_pretrained(d, trust_remote_code=True))


def test_generation_feeds_the_whole_decoder_sequence(model) -> None:  # type: ignore[no-untyped-def]
    import torch

    ids = torch.zeros((1, 7), dtype=torch.long, device="meta")
    inputs = model.prepare_inputs_for_generation(
        ids, next_sequence_length=1, past_key_values=object(), encoder_outputs=object()
    )
    assert tuple(inputs["decoder_input_ids"].shape) == (1, 7)  # not sliced to the newest token
    assert inputs.get("past_key_values") is None
    assert inputs["use_cache"] is False


def test_generation_config_defaults_to_no_cache(model) -> None:  # type: ignore[no-untyped-def]
    assert model.generation_config.use_cache is False


def test_override_keeps_the_parent_explicit_parameters(model) -> None:  # type: ignore[no-untyped-def]
    """generate() validates model_kwargs against this signature: dropping a parameter rejects caller inputs."""
    import inspect

    from transformers.generation.utils import GenerationMixin

    ours = inspect.signature(type(model).prepare_inputs_for_generation).parameters
    parent = inspect.signature(GenerationMixin.prepare_inputs_for_generation).parameters
    assert [n for n in parent if n not in ours] == []


def test_generate_accepts_everything_the_processor_returns(model) -> None:  # type: ignore[no-untyped-def]
    """Golden gate (b) on 26.08-4: `generate(**processor(...))` failed with "model_kwargs are not used by the
    model: ['attention_mask']" because the 0003 override dropped the parent's `attention_mask` parameter."""
    if not PREPARED_V11.is_dir():
        pytest.skip("prepared Nemotron-Parse v1.1 dir not found")
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(PREPARED_V11, trust_remote_code=True)
    inputs = processor(
        images=[Image.new("RGB", (640, 480), "white")],
        text=["</s><s><predict_bbox><predict_classes><output_markdown>"],
        return_tensors="pt",
        add_special_tokens=False,
    )
    assert "attention_mask" in inputs
    model._validate_model_kwargs(dict(inputs))  # what generate(**inputs) validates first
