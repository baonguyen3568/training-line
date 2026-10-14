"""Local datasets ("Thư mục của tôi", M1b B-3b): normalized records shared by data, train and eval (research spec).

A user folder holds images + ``labels.jsonl``; each line ``{"image": "<file in the folder>", "target": <str|object>,
"prompt"?: str}``. The ``data.local_folder`` node checks it and writes ``records.jsonl`` (one normalized record per
line, with the image's sha256). The model-specific text is produced only by the train worker
(``conversations_for``), because a data node does not know which model will read it:
- Nemotron-Parse: an object target becomes ``json2token(obj, sort_json_key=True)`` exactly like AutoModel's
  ``make_cord_v2_dataset`` (so token2json / field F1 work); a string target is used verbatim; ``prompt`` is ignored
  (the Parse collate uses its fixed task prompt).
- Chat VLMs: an object target is its JSON text; ``prompt`` defaults to ``DEFAULT_PROMPT``.
The train/val split of a local dataset is a stable hash of (seed, image name), so it is reproducible from run.json.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_PROMPT = "Describe this image."  # same as make_cord_v2_dataset: train and eval see the same prompt
Family = Literal["nemotron_parse", "chat"]


def image_name_ok(name: str) -> bool:
    r"""A plain relative POSIX path: no leading "/" or "~", no "\\", no empty / "." / ".." part, no control char.

    The one rule for image names: the folder scan (data.local_folder) and every records.jsonl read use it.
    """
    parts = name.split("/")
    return not (
        not name
        or len(name) > 512
        or name.startswith(("/", "~"))
        or "\\" in name
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
    )


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(min_length=1, max_length=512)  # file name relative to the dataset folder
    target: Any
    prompt: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Provenance of a machine-made target ("<catalog model id>@<revision>", data.caption_vlm); None = from the user's
    # labels. Kept per record so mix/filter never lose it: licence, "human labels only" eval sets (bao-security).
    generated_by: str | None = Field(None, max_length=200)

    @field_validator("image")
    @classmethod
    def _relative_inside(cls, value: str) -> str:
        """A plain relative POSIX path inside the dataset folder (bao-security R1: records.jsonl may be tampered with).

        Callers still check ``(folder / image).resolve()`` stays in the folder (a link inside it could point out).
        """
        if not image_name_ok(value):
            raise ValueError("image must be a relative path inside the dataset folder")
        return value


def _dump(record: Record) -> dict[str, Any]:
    """Record as written; ``generated_by`` only when set, so user-labelled data keeps its bytes (and its sha256)."""
    return record.model_dump(exclude={"generated_by"} if record.generated_by is None else set())


def write_records(path: Path, records: Iterable[Record]) -> str:
    """Write records.jsonl; returns its sha256 (part of the dataset's identity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(_dump(r), ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def image_path(folder: Path, record: Record) -> Path:
    """Host path of ``record.image``; ValueError if it resolves outside ``folder`` (e.g. through a link)."""
    root = folder.resolve()
    path = (root / record.image).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{record.image!r} resolves outside the dataset folder")
    return path


def read_records(path: Path) -> Iterator[Record]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Record.model_validate_json(line)


def in_validation(image: str, *, seed: int, val_fraction: float) -> bool:
    """Stable membership: the same name + seed is always on the same side, whatever else is in the folder."""
    digest = hashlib.sha256(f"{seed}:{image}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_fraction


def select(records: Iterable[Record], *, part: str, seed: int, val_fraction: float, limit: int = 0) -> list[Record]:
    """``part``: all | train | validation; ``limit`` keeps the first N of that part (0 = all)."""
    chosen = [
        r for r in records
        if part == "all" or in_validation(r.image, seed=seed, val_fraction=val_fraction) == (part == "validation")
    ]  # fmt: skip
    return chosen[:limit] if limit else chosen


def json2token(obj: Any, sort_json_key: bool = True) -> str:
    """Donut-style target text, identical to nemo_automodel.components.datasets.vlm.utils.json2token (26.08)."""
    if type(obj) is dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return str(obj["text_sequence"])
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        return "".join(rf"<s_{k}>" + json2token(obj[k], sort_json_key) + rf"</s_{k}>" for k in keys)
    if type(obj) is list:
        return r"<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def target_text(target: Any, family: Family) -> str:
    if isinstance(target, str):
        return target
    if family == "nemotron_parse":
        return json2token(target, sort_json_key=True)
    return json.dumps(target, ensure_ascii=False)


def asked(record: Record, family: Family) -> str:
    """The question a training sample is asked (what ``conversations_for`` writes)."""
    return DEFAULT_PROMPT if family == "nemotron_parse" or not record.prompt else record.prompt


MAX_PROMPT_CHARS = 2000  # = data.local_folder.MAX_PROMPT_CHARS = RunManifest.train_prompt's cap


def single_prompt(records: Iterable[Record], family: Family) -> str | None:
    """The one question every sample is asked (H-prompt: model card, trial), or None when they differ / no sample.

    A question over ``MAX_PROMPT_CHARS`` is None too, never cut (a cut question is not the one trained; security).
    """
    prompts = {asked(r, family) for r in records}
    prompt = prompts.pop() if len(prompts) == 1 else None
    return prompt if prompt is not None and len(prompt) <= MAX_PROMPT_CHARS else None


def conversations_for(records: Iterable[Record], family: Family) -> Iterator[dict[str, Any]]:
    """ShareGPT lines for AutoModel's ``make_meta_dataset`` (images relative to the meta's ``media_dir``)."""
    for r in records:
        prompt = asked(r, family)
        yield {
            "messages": [
                {"role": "user", "content": f"<image>\n{prompt}"},
                {"role": "assistant", "content": target_text(r.target, family)},
            ],
            "images": [r.image],
        }


def write_meta_dataset(out_dir: Path, name: str, records: Iterable[Record], family: Family, media_dir: str) -> Path:
    """``<out_dir>/<name>.jsonl`` + ``<out_dir>/<name>.meta.json`` (``make_meta_dataset`` input); returns the meta path.

    ``media_dir`` is the dataset folder as the job sees it (container path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = out_dir / f"{name}.jsonl"
    data.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in conversations_for(records, family)), encoding="utf-8"
    )
    meta = out_dir / f"{name}.meta.json"
    meta.write_text(json.dumps({name: {"file_name": data.name, "media_dir": media_dir}}, indent=2) + "\n")
    return meta


__all__ = [
    "DEFAULT_PROMPT",
    "MAX_PROMPT_CHARS",
    "Record",
    "asked",
    "conversations_for",
    "image_name_ok",
    "image_path",
    "in_validation",
    "json2token",
    "read_records",
    "select",
    "single_prompt",
    "target_text",
    "write_meta_dataset",
    "write_records",
]
