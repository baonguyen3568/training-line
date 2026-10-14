"""A Hugging Face dataset the user picks (ADR-021, P021 H4): the host side of an inspection.

The resolve job (egress) pins the revision, lists the files and fetches ONE bounded sample of the chosen split; then a
``data`` job (``--network none``) reads that sample (``download.inspect``) and writes our JSON + PNG thumbnails.
This module only reads what those jobs wrote: ``resolve.json``'s dataset block and ``inspect.json``, both validated
(types, sizes, closed sets), and the thumbnails (regular PNG files, size-capped). Nothing from the dataset is parsed
in the app process.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vlm_pipeline.common.download_rules import name_ok
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.runinfo import read_small_json
from vlm_pipeline.download.resolver import RepoFilesInvalidError, user_dataset_id

INSPECT_MAX = 2 << 20
THUMB_MAX = 512 * 1024
_PNG = b"\x89PNG\r\n\x1a\n"
_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
Role = Literal["image", "question", "answer", "messages"]
Kind = Literal["image", "messages", "text", "number", "list", "other"]


class DatasetHasScriptError(VlmPipelineError):
    """The dataset is only a loading script: the app never runs one (DATA-009)."""

    code: ClassVar[str] = "DATA-009"


class DatasetUnreadableError(VlmPipelineError):
    """No data file, or a sample that cannot be read (DATA-012, reason no_data_files | unreadable_sample | ...)."""

    code: ClassVar[str] = "DATA-012"


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
    format: Literal["parquet", "arrow", "jsonl", "json", "csv", "imagefolder", "webdataset", "unknown"]
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
        raise DatasetUnreadableError("no data file", params={"repo": repo, "reason": "no_data_files"})
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


# --------------------------------------------------------------------------------------------- the record (H5)
STORE_REL = Path(".cache") / "user_assets" / "datasets"
RECORD_MAX = 512 * 1024
DATASET_ID = re.compile(r"^user--ds--[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*$")
MAX_ID = 300
_COLUMN = r"^[A-Za-z0-9_.\- ]{1,64}$"


class ColumnMap(BaseModel):
    """Which column plays which role (the column-mapping screen): an image, and messages or question + answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(pattern=_COLUMN)
    question: str | None = Field(None, pattern=_COLUMN)
    answer: str | None = Field(None, pattern=_COLUMN)
    messages: str | None = Field(None, pattern=_COLUMN)

    @model_validator(mode="after")
    def _texts(self) -> Self:
        if self.messages is None and self.answer is None:
            raise ValueError("a messages column or an answer column is needed")
        if self.messages is not None and (self.question or self.answer):
            raise ValueError("messages, or question + answer: not both")
        return self

    def roles(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class _SplitFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    size: int = Field(ge=0)


class UserDatasetRecord(BaseModel):
    """One config + split of a pinned HF dataset revision the user added, with its column mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    id: str = Field(max_length=MAX_ID)
    repo: str = Field(max_length=193)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config: str = Field(pattern=_NAME)
    split: str = Field(pattern=_NAME)
    format: Literal["parquet", "arrow", "jsonl", "json", "csv", "imagefolder", "webdataset"]
    files: tuple[_SplitFile, ...] = Field(min_length=1, max_length=2000)
    license: str = Field(max_length=80)
    gated: bool
    columns: ColumnMap
    added_at: float
    license_confirmed: bool = False
    license_confirmed_sha: str | None = Field(None, pattern=r"^[0-9a-f]{40}$")
    rows: int | None = Field(None, ge=0)  # once converted

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.id != user_dataset_id(self.repo, self.revision, self.config, self.split):
            raise ValueError("id does not match repo + revision + config + split")
        if not all(name_ok(f.name) for f in self.files):
            raise ValueError("a split file name the download rules refuse")
        return self

    @property
    def size_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def to_api(self) -> dict[str, object]:
        return {"id": self.id, "name": f"{self.repo.split('/', 1)[1]} ({self.config}/{self.split})",
                "repo": self.repo, "revision": self.revision, "config": self.config, "split": self.split,
                "format": self.format, "size_gb": round(self.size_bytes / (1 << 30), 2), "license": self.license,
                "license_confirmed": self.license_confirmed, "gated": self.gated, "columns": self.columns.roles(),
                "rows": self.rows, "verified": False}  # fmt: skip


def _store(root: Path) -> Path:
    folder = root / STORE_REL
    if any(p.is_symlink() for p in (folder.parent, folder)):
        raise OSError("the user asset store must not be a symlink")
    return folder


def save(root: Path, record: UserDatasetRecord) -> None:
    folder = _store(root)
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(record.model_dump_json(indent=1))
        Path(tmp).replace(folder / f"{record.id}.json")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(root: Path, dataset_id: str) -> UserDatasetRecord | None:
    """The record, or None when absent or not valid (a hand-edited file never becomes a dataset)."""
    if len(dataset_id) > MAX_ID or not DATASET_ID.fullmatch(dataset_id):
        return None
    try:
        return UserDatasetRecord.model_validate(read_small_json(_store(root) / f"{dataset_id}.json", RECORD_MAX))
    except (OSError, ValueError, ValidationError):
        return None


def list_all(root: Path) -> list[UserDatasetRecord]:
    folder = root / STORE_REL
    names = sorted(p.stem for p in folder.glob("user--ds--*.json")) if folder.is_dir() else []
    return [r for r in (load(root, n) for n in names) if r is not None]


def delete(root: Path, dataset_id: str) -> bool:
    """The record and its converted folder (both in our own store; never through a link)."""
    import shutil

    if len(dataset_id) > MAX_ID or not DATASET_ID.fullmatch(dataset_id):
        return False
    folder = _store(root)
    path = folder / f"{dataset_id}.json"
    existed = path.is_file()
    path.unlink(missing_ok=True)
    data = folder / dataset_id
    if data.is_dir() and not data.is_symlink():
        shutil.rmtree(data)
    return existed


def confirm_license(root: Path, dataset_id: str) -> UserDatasetRecord | None:
    record = load(root, dataset_id)
    if record is None:
        return None
    confirmed = record.model_copy(update={"license_confirmed": True, "license_confirmed_sha": record.revision})
    save(root, confirmed)
    return confirmed


def data_folder(root: Path, dataset_id: str) -> Path:
    """Where the converted split lives (images + labels.jsonl: what data.local_folder reads)."""
    return root / STORE_REL / dataset_id


def from_inspection(run_dir: Path, repo: str, columns: ColumnMap, available: list[str]) -> UserDatasetRecord:
    """The record of an inspected split, its files from the resolve (never from the client)."""
    from vlm_pipeline.app.user_assets import read_resolve
    from vlm_pipeline.download.dataset_layout import layout

    meta = read_resolve(run_dir)
    block = dataset_block(repo, meta.dataset)
    if block.config is None or block.split is None or block.format == "unknown":
        raise DatasetUnreadableError("no data file", params={"repo": repo, "reason": "no_data_files"})
    missing = [c for c in columns.roles().values() if c not in available]
    if missing:
        raise DatasetColumnsMissingError("a mapped column is not in the dataset", params={
            "repo": repo, "missing": "image" if columns.image in missing else "answer",
            "available": ", ".join(available[:10])})  # fmt: skip
    sizes = {f.name: f.size for f in meta.files}
    names = [f.name for f in layout(sizes)[block.config][block.split]]
    if block.format == "imagefolder":  # the split's images live next to its metadata
        folders = {n.rsplit("/", 1)[0] + "/" for n in names if "/" in n}
        names += sorted(n for n in sizes if any(n.startswith(f) for f in folders) and n not in names)
    files = tuple(_SplitFile(name=n, size=sizes[n]) for n in sorted(set(names)))
    return UserDatasetRecord(
        id=user_dataset_id(repo, meta.sha, block.config, block.split), repo=repo, revision=meta.sha,
        config=block.config, split=block.split, format=block.format, files=files,
        license=(meta.license or "unknown")[:80], gated=meta.gated, columns=columns, added_at=time.time(),
    )  # fmt: skip


_LABEL_NAME = re.compile(r"^images/[0-9]{1,7}\.(png|jpg)$")


def _check_converted(out: Path) -> int:
    """The convert job's folder: labels.jsonl + images/<n>.png|jpg + report.json, regular files only; its rows."""
    report = read_small_json(out / "report.json", 64 * 1024)
    if not isinstance(report, dict) or "error" in report:
        reason = report.get("error") if isinstance(report, dict) else None
        raise DatasetUnreadableError("the split cannot be converted", params={
            "repo": "", "reason": "unreadable_sample" if reason != "over_budget" else "sample_too_large",
            "limit_mb": 0})  # fmt: skip
    rows = report.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise DatasetUnreadableError("bad report", params={"repo": "", "reason": "unreadable_sample"})
    for path in out.rglob("*"):
        rel = path.relative_to(out).as_posix()
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode) and rel == "images":
            continue
        if not stat.S_ISREG(st.st_mode) or not (rel in ("labels.jsonl", "report.json") or _LABEL_NAME.match(rel)):
            raise DatasetUnreadableError("unexpected file", params={"repo": "", "reason": "unreadable_sample"})
    return rows


def convert_and_install(service: Any, d: Any, record: UserDatasetRecord, files: Path, budget: int) -> int:
    """Run the convert ``data`` job on the verified files, check its folder, install it (all or none); rows."""
    import json
    import shutil

    from vlm_pipeline.common.jobrun import CONTAINER_ROOT

    name = f"{d.run_id}-cv"
    run_dir = service.root / "outputs" / "runs" / name
    out = run_dir / "out"

    def container(path: Path) -> str:
        return f"{CONTAINER_ROOT}/{path.relative_to(service.root).as_posix()}"

    spec_path = service.root / "outputs" / "runs" / f"{d.run_id}" / "convert_spec.json"
    spec_path.write_text(json.dumps({"files": [f.name for f in record.files if not f.name.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp"))], "format": record.format, "mapping": record.columns.roles(),
        "limit": 0, "max_bytes": budget * 3}), encoding="utf-8")  # fmt: skip
    argv = ("python", "-m", "vlm_pipeline.download.convert", "--files", container(files),
            "--spec", container(spec_path), "--out", container(out))  # fmt: skip
    final = service.run_data_job(d, name, argv, (files, spec_path.parent))
    if not (out / "report.json").is_file():
        raise DatasetUnreadableError("convert job failed", params={"repo": record.repo, "reason": "unreadable_sample"})
    try:
        rows = _check_converted(out)
    except DatasetUnreadableError as exc:
        raise DatasetUnreadableError(str(exc), params={**exc.params, "repo": record.repo}) from None
    if getattr(final, "exit_code", 0) not in (0, None) or rows == 0:
        raise DatasetUnreadableError("no usable row", params={"repo": record.repo, "reason": "unreadable_sample"})
    target = data_folder(service.root, record.id)
    if target.exists() or target.is_symlink():
        raise OSError("the dataset folder already exists")
    tmp = target.with_name(f".{record.id}.{os.getpid()}.tmp")
    shutil.copytree(out, tmp, symlinks=True)
    tmp.rename(target)
    return rows


__all__ = [
    "ColumnMap",
    "DatasetBlock",
    "DatasetColumnsMissingError",
    "DatasetHasScriptError",
    "DatasetUnreadableError",
    "InspectResult",
    "UserDatasetRecord",
    "api_result",
    "confirm_license",
    "convert_and_install",
    "data_folder",
    "dataset_block",
    "delete",
    "from_inspection",
    "list_all",
    "load",
    "read_inspect",
    "save",
    "thumb_path",
]
