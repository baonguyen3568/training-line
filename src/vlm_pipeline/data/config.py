"""Parameters of the data engine, shared by the data nodes and the data workers (single source, P009 §2.2).

Host-safe: pydantic only. UI hints live on each field under ``x-ui`` (see ``vlm_pipeline.nodes.spec.ui``).
Defaults and ranges: .team/knowledge/node-params-v1.md §1.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from vlm_pipeline.common.params import LocalPath, Params, ui

DEFAULT_SEED = 1234

# Hub datasets vetted by security (configs/security/licenses.yaml). Pinned revisions are resolved by the worker.
HubDataset = Literal["naver-clova-ix/cord-v2"]
# 0 = all samples; the others are the "quick run" sizes.
SampleLimit = Literal[0, 100, 500, 1000]


class HubDatasetParams(Params):
    repo: HubDataset = Field("naver-clova-ix/cord-v2", json_schema_extra=ui(order=0, summary=True))
    max_samples: SampleLimit = Field(0, json_schema_extra=ui(order=1, option_keys="sample_limit"))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))


# Label file formats data.local_folder normalises at load (research 2026-10-13: LLaVA-Instruct JSON, ShareGPT as in
# LLaMA-Factory dataset_info, HF chat messages with {"type": "image"}); auto = detected from the first records.
InputFormat = Literal["auto", "llava", "sharegpt", "hf_messages"]


class LocalFolderParams(Params):
    path: LocalPath = Field(
        "data/sample-invoices",
        description="Folder inside the project with images and labels.jsonl",
        json_schema_extra=ui(order=0, summary=True),
    )
    max_samples: SampleLimit = Field(0, json_schema_extra=ui(order=1, option_keys="sample_limit"))
    max_image_side_px: int = Field(4096, ge=512, le=8192, json_schema_extra=ui(advanced=True))
    input_format: InputFormat = Field("auto", json_schema_extra=ui(advanced=True, option_keys="input_format"))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))


class SplitParams(Params):
    val_fraction: float = Field(0.1, ge=0.05, le=0.3, json_schema_extra=ui(widget="slider", order=0))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))


# --------------------------------------------------------------------------------- curation nodes (Release 1.0, B)
# Defaults and ranges: bao-research 2026-10-13 (image limits = common/image_rules.py; P4-2 data_quality standards).
# CPU nodes run on the host (Pillow + hashlib): no new dependency, images opened through image_rules only.


class FilterParams(Params):
    min_short_side_px: int = Field(64, ge=16, le=1024, json_schema_extra=ui(order=0, summary=True))
    max_target_chars: int = Field(8000, ge=100, le=100_000, json_schema_extra=ui(order=1))
    drop_broken: bool = Field(True, json_schema_extra=ui(order=2))
    max_long_side_px: int = Field(4096, ge=512, le=8192, json_schema_extra=ui(advanced=True))
    max_megapixels: float = Field(40.0, ge=1, le=40, json_schema_extra=ui(advanced=True))
    min_target_chars: int = Field(1, ge=0, le=1000, json_schema_extra=ui(advanced=True))

    @model_validator(mode="after")
    def _bounds_in_order(self) -> Self:
        if self.min_target_chars >= self.max_target_chars:
            raise ValueError("min_target_chars must be below max_target_chars")
        return self


DedupeMethod = Literal["sha256", "dhash"]


_DHASH_ONLY = {"compare": ["dhash"]}


class DedupeParams(Params):
    compare: DedupeMethod = Field("sha256", json_schema_extra=ui(order=0, summary=True, option_keys="dedupe_method"))
    check_leak: bool = Field(True, json_schema_extra=ui(order=1))  # drop train samples whose image is in heldout
    dhash_max_distance: int = Field(4, ge=0, le=16, json_schema_extra=ui(advanced=True, visible_when=_DHASH_ONLY))


MixStrategy = Literal["concat", "interleave_under", "interleave_over"]
_INTERLEAVE = {"strategy": ["interleave_under", "interleave_over"]}  # weights mean nothing for concat


class MixParams(Params):
    strategy: MixStrategy = Field("concat", json_schema_extra=ui(order=0, summary=True, option_keys="mix_strategy"))
    weight_1: float = Field(1.0, ge=0, le=10, json_schema_extra=ui(order=1, visible_when=_INTERLEAVE))  # shares
    weight_2: float = Field(1.0, ge=0, le=10, json_schema_extra=ui(order=2, visible_when=_INTERLEAVE))
    weight_3: float = Field(1.0, ge=0, le=10, json_schema_extra=ui(advanced=True, visible_when=_INTERLEAVE))
    weight_4: float = Field(1.0, ge=0, le=10, json_schema_extra=ui(advanced=True, visible_when=_INTERLEAVE))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))
