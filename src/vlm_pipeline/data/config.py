"""Parameters of the data engine, shared by the data nodes and the data workers (single source, P009 §2.2).

Host-safe: pydantic only. UI hints live on each field under ``x-ui`` (see ``vlm_pipeline.nodes.spec.ui``).
Defaults and ranges: .team/knowledge/node-params-v1.md §1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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


class LocalFolderParams(Params):
    path: LocalPath = Field(
        "data/sample-invoices",
        description="Folder inside the project with images and labels.jsonl",
        json_schema_extra=ui(order=0, summary=True),
    )
    max_samples: SampleLimit = Field(0, json_schema_extra=ui(order=1, option_keys="sample_limit"))
    max_image_side_px: int = Field(4096, ge=512, le=8192, json_schema_extra=ui(advanced=True))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))


class SplitParams(Params):
    val_fraction: float = Field(0.1, ge=0.05, le=0.3, json_schema_extra=ui(widget="slider", order=0))
    seed: int = Field(DEFAULT_SEED, json_schema_extra=ui(advanced=True))
