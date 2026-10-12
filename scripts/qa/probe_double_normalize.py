"""Double-normalisation probe for Nemotron-Parse on the native AutoModel class (runs INSIDE a vlm-automodel image).

A constant white image is normalised exactly as the processor would (``preprocessor_config.json``: rescale then
mean/std), fed to ``RadioWithNeck`` built from the model config with random weights (normalisation does not depend
on weights), and the input of RADIO's ``patch_generator`` is captured with a forward-pre-hook. CPU only, offline.
RADIO's input conditioner casts to bf16, so expectations are compared after bf16 rounding.

Usage (from the repo root, no GPU):
  docker run --rm --network none --user $(id -u):$(id -g) -e HF_HUB_OFFLINE=1 -e HOME=/tmp \
    -v $PWD:/workspace:ro --entrypoint python vlm-automodel:<tag> \
    /workspace/scripts/qa/probe_double_normalize.py /workspace/.cache/prepared/<model>/model
Prints one JSON line with ``verdict`` = single | double | other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def probe(model_dir: str) -> dict[str, Any]:
    import numpy as np
    import torch
    from nemo_automodel.components.models.nemotron_parse.model import NemotronParseConfig, RadioWithNeck

    def bf16(values: Any) -> Any:
        return torch.tensor(values, dtype=torch.float32).bfloat16().float().numpy()

    cfg = NemotronParseConfig.from_pretrained(model_dir)
    pp = json.loads((Path(model_dir) / "preprocessor_config.json").read_text())
    mean, std = np.array(pp["image_mean"]), np.array(pp["image_std"])
    white = 255 * pp["rescale_factor"] if pp.get("do_rescale") else 255.0
    pixel = (white - mean) / std if pp.get("do_normalize") else np.full(3, white)
    pixel_values = torch.tensor(pixel, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 256, 256).contiguous()

    torch.manual_seed(0)
    encoder = RadioWithNeck(cfg.encoder).float().eval()
    captured: dict[str, Any] = {}
    target = next(m for name, m in encoder.named_modules() if name.endswith("patch_generator"))
    target.register_forward_pre_hook(lambda _m, args: captured.setdefault("x", args[0].detach().float()))
    with torch.no_grad():
        try:
            encoder(pixel_values)
        except RuntimeError as exc:  # the neck may reject a small image; the hook has already fired
            captured.setdefault("error", repr(exc)[:200])

    x = captured["x"][0, :, 128, 128].numpy()
    single = (1 - mean) / std
    double = (single - mean) / std
    d_single = float(np.abs(x - bf16(single)).max())
    d_double = float(np.abs(x - bf16(double)).max())
    verdict = "single" if d_single <= 1e-3 else "double" if d_double <= 1e-3 else "other"
    return {
        "encoder_has_flag": hasattr(cfg.encoder, "image_processor_normalizes"),
        "model_flag": encoder.image_processor_normalizes,
        "processor_do_normalize": bool(pp.get("do_normalize")),
        "processor_pixel": pixel.round(4).tolist(),
        "patch_embed_input": x.round(4).tolist(),
        "expected_single": single.round(4).tolist(),
        "expected_double": double.round(4).tolist(),
        "d_single": d_single,
        "d_double": d_double,
        "verdict": verdict,
        "note": captured.get("error"),
    }


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1])))
