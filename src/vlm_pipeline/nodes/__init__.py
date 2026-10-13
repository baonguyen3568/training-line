"""First-party nodes and the explicit v1 registry (ADR-004 A3: no plugin discovery for first-party nodes)."""

from __future__ import annotations

from vlm_pipeline.nodes.data import DATA_SPLIT
from vlm_pipeline.nodes.evaluate import BENCH_PERF, EVAL_BENCHMARKS
from vlm_pipeline.nodes.model_ops import EXPORT_HF_DIR, QUANTIZE_MODELOPT_PTQ, SERVE_VLLM, TRAIN_MERGE_LORA
from vlm_pipeline.nodes.rules import RULES
from vlm_pipeline.nodes.source import DATA_HF_LOAD, DATA_LOCAL_FOLDER, MODEL_HF_PICK
from vlm_pipeline.nodes.spec import Registry
from vlm_pipeline.nodes.train import TRAIN_LORA_VLM, TRAIN_SFT_VLM

# Container images are not pinned per node yet (M1 runs with the FakeWorker first).
REGISTRY = Registry(
    [
        # Sources
        MODEL_HF_PICK,
        DATA_HF_LOAD,
        DATA_LOCAL_FOLDER,
        # Data
        DATA_SPLIT,
        # Train
        TRAIN_LORA_VLM,
        TRAIN_SFT_VLM,
        TRAIN_MERGE_LORA,
        # Test, quantize, evaluate, export
        SERVE_VLLM,
        QUANTIZE_MODELOPT_PTQ,
        EVAL_BENCHMARKS,
        BENCH_PERF,
        EXPORT_HF_DIR,
    ],
    rules=RULES,
    allow_unpinned=True,
)

__all__ = ["REGISTRY"]
