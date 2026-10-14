"""A Hugging Face dataset the user picks (ADR-021, P021 H4): the host side of an inspection.

The resolve job (egress) pins the revision, lists the files and fetches ONE bounded sample of the chosen split; then a
``data`` job (``--network none``) reads that sample (``download.inspect``) and writes our JSON + PNG thumbnails.
This module only reads what those jobs wrote: ``resolve.json``'s dataset block and ``inspect.json``, both validated
(types, sizes, closed sets), and the thumbnails (regular PNG files, size-capped). Nothing from the dataset is parsed
in the app process.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.runinfo import read_small_json
from vlm_pipeline.download.resolver import RepoFilesInvalidError

INSPECT_MAX = 2 << 20
THUMB_MAX = 512 * 1024
_PNG = b"\x89PNG\r\n\x1a\n"
_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
Role = Literal["image", "question", "answer", "messages"]
Kind = Literal["image", "messages", "text", "number", "list", "other"]


class DatasetHasScriptError(VlmPipelineError):
    """The dataset is only a loading script: the app never runs one (DATA-009)."""

    code: ClassVar[str] = "DATA-009"


class DatasetColumnsMissingError(VlmPipelineError):
    """No column the app can use was found in the sample (DATA-010)."""

    code: ClassVar[str] = "DATA-010"


class DatasetBlock(BaseModel):
    """The ``dataset`` block of ``resolve.json`` (download.hub._dataset_choice)."""

    model_config = ConfigDict(extra="forbid")

    configs: list[str] = Field(max_length=200)
    has_script: bool
    has_parquet_export: bool
    config: str | None = Field(None, pattern=_NAME)
    split: str | None = Field(None, pattern=_NAME)
    splits: list[str] = Field(max_length=50)
    format: Literal["parquet", "arrow", "jsonl", "json", "csv", "imagefolder", "unknown"]
    sample: list[str] = Field(max_length=10)
    split_bytes: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _names(self) -> Self:
        if not all(re.fullmatch(_NAME, n) for n in [*self.configs, *self.splits]):
            raise ValueError("a config or split name outside the allowed form")
        return self


class _Column(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.\- ]{1,64}$")
    kind: Kind


class _Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(None, max_length=500)
    answer: str | None = Field(None, max_length=500)
    thumb: int | None = Field(None, ge=0, lt=20)
    error: Literal["image_missing", "image_not_a_file", "image_too_large", "image_too_many_pixels",
                   "image_unreadable"] | None = None  # fmt: skip


class InspectResult(BaseModel):
    """``inspect.json`` as the data job wrote it: untrusted, so typed and bounded."""

    model_config = ConfigDict(extra="forbid")

    columns: list[_Column] = Field(max_length=64)
    suggested: dict[Role, str]
    samples: list[_Sample] = Field(max_length=20)
    rows_read: int = Field(ge=0, le=20)
    error: Literal["no_sample", "unreadable"] | None = None

    @model_validator(mode="after")
    def _suggested_columns_exist(self) -> Self:
        if not set(self.suggested.values()) <= {c.name for c in self.columns}:
            raise ValueError("a suggested column that is not a column")
        return self


def read_inspect(out: Path) -> InspectResult:
    try:
        return InspectResult.model_validate(read_small_json(out / "inspect.json", INSPECT_MAX))
    except (OSError, ValueError, ValidationError) as exc:
        raise RepoFilesInvalidError(f"inspect.json: {type(exc).__name__}") from None


def thumb_path(out: Path, n: int) -> Path | None:
    """A thumbnail the job wrote, only when it is a small regular PNG file (never a link)."""
    path = out / "thumbs" / f"{n}.png"
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > THUMB_MAX:
            return None
        with path.open("rb") as fh:
            return path if fh.read(len(_PNG)) == _PNG else None
    except OSError:
        return None


def dataset_block(repo: str, raw: Any) -> DatasetBlock:
    """The dataset block, or the catalog error a loading-script-only repo deserves."""
    try:
        block = DatasetBlock.model_validate(raw)
    except ValidationError:
        raise RepoFilesInvalidError("resolve.json: dataset") from None
    if block.config is None:  # no data file the hub layouts describe
        if block.has_script:
            raise DatasetHasScriptError("only a loading script", params={
                "repo": repo, "has_parquet_export": block.has_parquet_export})  # fmt: skip
        raise DatasetColumnsMissingError("no data file", params={"repo": repo, "missing": "image", "available": ""})
    return block


def api_result(repo: str, sha: str, block: DatasetBlock, inspected: InspectResult, thumb_url: str) -> dict[str, Any]:
    """The ``result`` of GET /api/user-assets/inspections/{id} (P021 §8)."""
    samples = []
    for s in inspected.samples:
        item: dict[str, Any] = {k: v for k, v in s.model_dump().items() if v is not None and k != "thumb"}
        if s.thumb is not None:
            item["thumb_url"] = f"{thumb_url}/{s.thumb}.png"
        samples.append(item)
    return {"repo": repo, "revision": sha, "configs": block.configs, "splits": block.splits, "config": block.config,
            "split": block.split, "format": block.format,
            "columns": [c.model_dump() for c in inspected.columns], "suggested": inspected.suggested,
            "samples": samples, "rows_estimate": None, "split_bytes": block.split_bytes,
            "has_script": block.has_script, "sample_error": inspected.error}  # fmt: skip


__all__ = [
    "DatasetBlock",
    "DatasetColumnsMissingError",
    "DatasetHasScriptError",
    "InspectResult",
    "api_result",
    "dataset_block",
    "read_inspect",
    "thumb_path",
]
