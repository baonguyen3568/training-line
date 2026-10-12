"""Regression tests for third_party/patches/automodel/nemotron_parse.patch (runs INSIDE vlm-automodel, offline).

Bugs (bao-fixcode, 2026-10-12; .team/knowledge/fixcode.md §2.6b):
1. ``NemotronParseConfig`` copies encoder keys only if ``RADIOConfig`` already has them, so the encoder
   normalize flag was dropped for EVERY Parse version -> RadioWithNeck re-normalized already-normalized pixels.
   v1.2/2.0 additionally renamed the flag ``image_processor_normalizes`` -> ``processor_normalizes``.
2. Parse 2.0 ships ``tie_word_embeddings: true`` and no ``lm_head.weight``; the class was ``UNTIED_ONLY``.

Must FAIL on vlm-automodel:26.08-1 and PASS on 26.08-2. Config/meta-device only: no weights, no GPU, no network.
RADIO code comes from the security-pinned local copy (C-RADIOv2-H@0d8f4c18), never from the Hub.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
# Host venv has no AutoModel: skip there; inside the image the module exists, so a missing fix still FAILS.
pytest.importorskip("nemo_automodel")

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "model_configs" / "nemotron_parse"
RADIO_DIR = Path(
    os.environ.get(
        "VLM_CRADIO_V2H_DIR",
        str(REPO / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "c-radiov2-h_0d8f4c18"),
    )
)
VERSIONS = ["v1.1", "v1.2", "v2.0"]


def _prepared_config_dir(version: str, tmp_path: Path) -> Path:
    """Copy the pinned config.json and point the encoder at the local pinned RADIO copy (offline)."""
    if not (RADIO_DIR / "config.json").is_file():
        pytest.skip(f"pinned C-RADIOv2-H copy not found at {RADIO_DIR}")
    cfg = json.loads((FIXTURES / f"config_{version}.json").read_text())
    enc = cfg["encoder"]
    enc["_name_or_path"] = str(RADIO_DIR)
    enc["auto_map"] = {
        "AutoConfig": f"{RADIO_DIR}--hf_model.RADIOConfig",
        "AutoModel": f"{RADIO_DIR}--hf_model.RADIOModel",
    }
    out = tmp_path / version
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg))
    return out


@pytest.mark.parametrize("version", VERSIONS)
def test_encoder_normalize_flag_survives_config(version: str, tmp_path: Path) -> None:
    from nemo_automodel.components.models.nemotron_parse.model import NemotronParseConfig

    cfg = NemotronParseConfig.from_pretrained(_prepared_config_dir(version, tmp_path), trust_remote_code=True)
    # All three checkpoints declare that their image processor normalizes (do_normalize: true).
    assert getattr(cfg.encoder, "image_processor_normalizes", None) is True


@pytest.mark.parametrize("version", VERSIONS)
def test_encoder_flag_survives_save_and_reload(version: str, tmp_path: Path) -> None:
    from nemo_automodel.components.models.nemotron_parse.model import NemotronParseConfig

    cfg = NemotronParseConfig.from_pretrained(_prepared_config_dir(version, tmp_path), trust_remote_code=True)
    saved = tmp_path / f"{version}-saved"
    cfg.save_pretrained(saved)
    again = NemotronParseConfig.from_pretrained(saved, trust_remote_code=True)
    assert again.encoder.image_processor_normalizes is True


@pytest.mark.parametrize(("version", "tied"), [("v1.1", False), ("v1.2", False), ("v2.0", True)])
def test_model_builds_with_correct_normalize_and_tie(version: str, tied: bool, tmp_path: Path) -> None:
    import torch
    from nemo_automodel.components.models.nemotron_parse.model import (
        NemotronParseConfig,
        NemotronParseForConditionalGeneration,
    )

    cfg = NemotronParseConfig.from_pretrained(_prepared_config_dir(version, tmp_path), trust_remote_code=True)
    with torch.device("meta"):
        model = NemotronParseForConditionalGeneration(cfg)
    assert model.encoder.image_processor_normalizes is True
    shares = model.lm_head.weight is model.decoder.embed_tokens.weight
    assert shares is tied, f"{version}: lm_head tied={shares}, expected {tied}"


def test_disagreeing_flag_names_new_name_wins_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Behaviour through the public config API (fails on 26.08-1 because the flag is dropped, not because a helper
    is missing). The preflight check (common/config_aliases.py) blocks such configs earlier; this is defence in depth.
    """
    from nemo_automodel.components.models.nemotron_parse.model import NemotronParseConfig

    d = _prepared_config_dir("v1.2", tmp_path)
    cfg_json = json.loads((d / "config.json").read_text())
    cfg_json["encoder"]["processor_normalizes"] = False
    cfg_json["encoder"]["image_processor_normalizes"] = True
    (d / "config.json").write_text(json.dumps(cfg_json))
    with caplog.at_level("WARNING"):
        cfg = NemotronParseConfig.from_pretrained(d, trust_remote_code=True)
    assert getattr(cfg.encoder, "image_processor_normalizes", None) is False
    assert "disagree" in caplog.text
