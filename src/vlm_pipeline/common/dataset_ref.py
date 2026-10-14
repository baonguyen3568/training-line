"""The ``Dataset`` artifact between graph nodes: a small JSON descriptor, not a copy of the data (M1b).

Written by the data workers (``data.hf_load``, ``data.split``, later ``data.local_folder``) and read by every consumer
(train, eval, quantize calibration). A "quick run" size or a split costs no copy: it is a field here. Paths are the
container paths the jobs see (``/workspace/...``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    repo: str | None = None  # hub repo id; None for local data
    revision: str | None = None  # pinned hub revision
    path: str  # container path of the dataset snapshot / folder
    split: str = "train"  # split of that dataset this artifact stands for (local data: all | train | validation)
    limit: int = Field(0, ge=0)  # first N samples (0 = all)
    # Local folder (data.local_folder): normalized records + the split rule (common.local_records), else None.
    records: str | None = None  # container path of records.jsonl (written by the data node)
    records_sha256: str | None = None  # identity of the content: same records = same data, whatever the path
    val_fraction: float | None = None
    seed: int | None = None
    # User-added HF datasets these records come from (ADR-021 H5d): local records, yet their licence must reach an
    # export. Carried through split / filter / dedupe / caption; part of the identity.
    origins: tuple[str, ...] = ()

    @property
    def hash(self) -> str:
        """Content identity for the cache key: same snapshot + split + limit = same data.

        ``repo@rev[#split=x][#limit=n]``: the train split keeps the pre-split format (existing cache keys stay valid),
        and everything after the revision follows ``#`` (RunManifest reads the revision as ``@<40 hex>(#|$)``).
        """
        if self.records_sha256:  # local data: content, split rule and size
            rule = "" if self.split == "all" else f"#val={self.val_fraction}#seed={self.seed}"
            return (
                f"local:{self.records_sha256}#split={self.split}{rule}"
                + (f"#limit={self.limit}" if self.limit else "")
                + (f"#origin={'+'.join(sorted(self.origins))}" if self.origins else "")
            )
        origin = f"{self.repo}@{self.revision}" if self.repo else self.path
        return (
            origin
            + (f"#split={self.split}" if self.split != "train" else "")
            + (f"#limit={self.limit}" if self.limit else "")
        )


def read_dataset_ref(path: Path) -> DatasetRef:
    """Descriptors written before ``split``/``schema_version`` existed read as the train split."""
    return DatasetRef.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_dataset_ref(path: Path, ref: DatasetRef) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["DatasetRef", "read_dataset_ref", "write_dataset_ref"]
