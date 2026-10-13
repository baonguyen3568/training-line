"""The ``Model`` artifact of ``model.hf_pick``: a small descriptor in the node's run dir, not the weights (M1b).

Node outputs must live inside their run dir (executor ``check_output_paths``: the cache index and retention only
ever see run dirs). A catalog model's weights live in the project cache (``.cache/hf/hub`` snapshot or the pinned
``.cache/prepared`` copy), which retention never touches, so the pick node writes ``model.json`` pointing there and
every consumer resolves it through ``RealWorker.model_dir``. Trained models are plain folders in their run dir and
need no descriptor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

FILE_NAME = "model.json"
CACHE_REL = Path(".cache")


class PickedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["vlm-model-ref"] = "vlm-model-ref"
    schema_version: Literal[1] = 1
    repo: str
    revision: str
    path: str  # project-relative folder under .cache/ (never absolute: the project may move or be mounted)


def write_model_ref(run_dir: Path, ref: PickedModel) -> Path:
    path = run_dir / FILE_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8")
    return path


def is_model_ref(path: Path) -> bool:
    return path.name == FILE_NAME and path.is_file()


def resolve_model_ref(path: Path, root: Path) -> Path:
    """The cached model folder a descriptor points at; refuses anything outside ``<root>/.cache``."""
    ref = PickedModel.model_validate(json.loads(path.read_text(encoding="utf-8")))
    cache = (root / CACHE_REL).resolve()
    target = (root / ref.path).resolve()
    if Path(ref.path).is_absolute() or not target.is_relative_to(cache) or not target.is_dir():
        raise ValueError(f"{path}: model folder must be an existing folder under .cache/")
    return target


__all__ = ["FILE_NAME", "PickedModel", "is_model_ref", "resolve_model_ref", "write_model_ref"]
