"""First-party nodes and the explicit v1 registry (ADR-004 A3: no plugin discovery for first-party nodes)."""

from __future__ import annotations

from vlm_pipeline.nodes.data import DATA_SPLIT
from vlm_pipeline.nodes.source import DATA_HF_LOAD, DATA_LOCAL_FOLDER, MODEL_HF_PICK
from vlm_pipeline.nodes.spec import Registry
from vlm_pipeline.nodes.train import TRAIN_LORA_VLM, TRAIN_SFT_VLM

# Container images are not pinned per node yet (M1 runs with the FakeWorker first).
REGISTRY = Registry(
    [MODEL_HF_PICK, DATA_HF_LOAD, DATA_LOCAL_FOLDER, DATA_SPLIT, TRAIN_LORA_VLM, TRAIN_SFT_VLM],
    allow_unpinned=True,
)

__all__ = ["REGISTRY"]
