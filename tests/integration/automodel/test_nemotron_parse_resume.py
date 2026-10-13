"""BUG-003 regression (patch 0005): a freshly built Nemotron-Parse must have the state_dict it saves after training.

RADIO's input conditioner (norm_mean/norm_std buffers) was swapped for nn.Identity at the FIRST FORWARD, so training
checkpoints lack those keys while a model built for resume still has them: DCP's strict load failed with "Missing key
in checkpoint state_dict: ...input_conditioner.norm_mean" (resume smoke 2026-10-13, run 20261013-0851-resume-b2-a93e).
Runs INSIDE vlm-automodel (offline, meta device, CPU). Must FAIL on 26.08-5, PASS patched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
CONDITIONER = "encoder.model_encoder.radio_model.input_conditioner."
BASE_WEIGHTS = REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model" / "model.safetensors"


def _build(tmp_path: Path, *, normalizes: bool) -> Any:
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
    cfg["encoder"]["image_processor_normalizes"] = normalizes
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with torch.device("meta"):
        return NemotronParseForConditionalGeneration(
            NemotronParseConfig.from_pretrained(tmp_path, trust_remote_code=True)
        )


def test_built_model_has_no_conditioner_buffers_when_the_processor_normalizes(tmp_path: Path) -> None:
    """The keys a resumed run asks the shard for must be the keys the trained model saved (no forward between)."""
    model = _build(tmp_path, normalizes=True)
    assert [k for k in model.state_dict() if k.startswith(CONDITIONER)] == []


def test_conditioner_is_kept_when_the_model_normalizes_itself(tmp_path: Path) -> None:
    """Without external normalization the conditioner is the only normalization: it must stay (and be saved)."""
    model = _build(tmp_path, normalizes=False)
    assert sorted(k for k in model.state_dict() if k.startswith(CONDITIONER)) == [
        CONDITIONER + "norm_mean",
        CONDITIONER + "norm_std",
    ]


def test_base_checkpoint_carries_the_conditioner_keys_the_patched_model_no_longer_requests() -> None:
    """Why the keys diverged: NVIDIA's file has them, so a fresh start loads a SUPERSET (extra keys are not requested);
    a resume loads the trained shard, which never had them."""
    if not BASE_WEIGHTS.is_file():
        pytest.skip("prepared Nemotron-Parse v1.1 weights not found")
    from vlm_pipeline.train.ckpt_complete import read_header

    header, _ = read_header(BASE_WEIGHTS)
    assert {CONDITIONER + "norm_mean", CONDITIONER + "norm_std"} <= header.keys()
