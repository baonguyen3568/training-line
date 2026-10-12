"""P-0 check (runs INSIDE vlm-automodel, --network none, HF offline, empty HF modules cache).

Builds Nemotron-Parse-v1.1 from the *prepared* config only (no weights, meta device) and proves that the RADIO
encoder code is resolved from the pinned local copy, never from ``nvidia/C-RADIOv2-H`` on the Hub. Also reports
whether AutoModel uses its native NemotronParse class or the repo's remote class. Prints one JSON line.
"""

import inspect
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

model_dir = sys.argv[1]
radio_dir = sys.argv[2]
out: dict[str, object] = {
    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    "modules_cache_empty_at_start": not (
        Path(os.environ.get("HF_MODULES_CACHE", "/nonexistent")).is_dir()
        and any(Path(os.environ["HF_MODULES_CACHE"]).iterdir())
    ),
}

cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
out["config_class"] = f"{type(cfg).__module__}.{type(cfg).__name__}"
enc_cls = type(cfg.encoder)
out["encoder_config_class"] = f"{enc_cls.__module__}.{enc_cls.__name__}"
out["encoder_config_file"] = inspect.getfile(enc_cls)

from nemo_automodel import NeMoAutoModelForImageTextToText  # noqa: E402

with torch.device("meta"):
    model = NeMoAutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
out["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
radio_mods = sorted(
    {f"{type(m).__module__}.{type(m).__name__}" for m in model.modules() if "radio" in type(m).__name__.lower()}
)
out["radio_classes"] = radio_mods
# Remote (dynamic-module) RADIO classes only; nemo_automodel's native RadioWithNeck wrapper is first-party code.
radio_files = sorted(
    {
        inspect.getfile(type(m))
        for m in model.modules()
        if "radio" in type(m).__name__.lower() and type(m).__module__.startswith("transformers_modules")
    }
)
out["radio_files"] = radio_files
# transformers names the dynamic module after the local dir, escaping "-" as "_hyphen_".
pinned_name = Path(radio_dir).name.replace("-", "_hyphen_")
bad = [f for f in [out["encoder_config_file"], *radio_files] if "nvidia" in f.split("transformers_modules")[-1]]
ok = bool(radio_files) and not bad and all(pinned_name in f for f in [out["encoder_config_file"], *radio_files])
out["radio_from_pinned_copy"] = ok
out["params_total_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
print(json.dumps(out))
sys.exit(0 if ok else 1)
