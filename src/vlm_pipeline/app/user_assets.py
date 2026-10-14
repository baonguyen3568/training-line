"""Models the user added from Hugging Face (ADR-021, P021 H2): the record store and the host side of a resolve.

A record holds the ``CatalogEntry`` the resolver built (tier user, security unreviewed), the licence confirmation and
the quick-trial run id. Records live under the data root (``.cache/user_assets/models/<id>.json``), never in
``configs/``: nobody reviews them, so every read validates them again and a bad file reads as "not added" (fail
closed). Writes are atomic (temp file + rename in the same directory).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vlm_pipeline.common.catalog import USER_MODELS_REL, CatalogEntry, normalize_license
from vlm_pipeline.common.download_rules import name_ok
from vlm_pipeline.common.errors import VlmPipelineError
from vlm_pipeline.common.paths import GIB
from vlm_pipeline.common.runinfo import read_small_json
from vlm_pipeline.download import resolver
from vlm_pipeline.download.hub import INDEX_NAME, meta_caps

logger = logging.getLogger(__name__)

STORE_REL = USER_MODELS_REL
RECORD_MAX = 64 * 1024
TEMPLATE_MAX = 256 * 1024
RESOLVE_MAX = 4 << 20  # resolve.json: up to 2 000 file names
MAX_ID = 206
USER_ID = re.compile(r"^user--[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)*$")  # = nodes.source.USER_MODEL_PATTERN, never ".."
DEFAULT_MAX_MODEL_GB = 40.0  # ADR-021 A1: per user-added model; Settings raises it (H5c)


class LicenseNotConfirmedError(VlmPipelineError):
    """No weight of a user-added model before its licence is confirmed for THIS revision (SEC-005, security §7)."""

    code: ClassVar[str] = "SEC-005"


class UserModelTooLargeError(ValueError):
    """Over the per-model cap (raised by the resolve finish; the download service maps it to NET-004)."""

    def __init__(self, size_gb: float, cap_gb: float) -> None:
        super().__init__("over the user model cap")
        self.params = {"size_gb": round(size_gb, 1), "quota_gb": round(cap_gb, 1), "reason": "model_cap"}


class UserModelRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    entry: CatalogEntry
    family: str = Field(pattern=r"^[a-z0-9_]+$")
    architecture: str = Field(max_length=resolver.MAX_NAME)
    added_at: float
    license_confirmed: bool = False
    license_confirmed_at: float | None = None
    license_confirmed_sha: str | None = Field(None, pattern=r"^[0-9a-f]{40}$")  # the revision the user agreed to
    # sha256 of <id>.train.yaml as rendered at add time: a recipe can name Python callables (_target_), so a template
    # edited on disk must never train (security H3); the record itself is validated on every read.
    template_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    license_card: str | None = Field(None, max_length=200)  # the card's own words (entry.license is normalized)
    trial_run_id: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    @model_validator(mode="after")
    def _a_user_entry(self) -> Self:
        e = self.entry
        if (e.tier, e.security, e.remote_code, e.ram_source) != ("user", "unreviewed", "none", "estimated"):
            raise ValueError("not a user-added entry")
        if e.id != resolver.user_model_id(e.repo, e.revision):
            raise ValueError("id does not match repo + revision")
        return self

    def to_api(self) -> dict[str, object]:
        from vlm_pipeline.common.obligations import load_obligations, model_obligations

        e = self.entry
        licence = normalize_license(e.license)  # read the same way as a dataset's (security)
        license_class, duties = model_obligations(e.model_copy(update={"license": licence}), load_obligations())
        return {"license_class": license_class, "obligations": duties,
                "id": e.id, "name": e.display_name, "repo": e.repo, "revision": e.revision, "family": self.family,
                "architecture": self.architecture, "params_b": e.params_b, "size_gb": e.disk_gb,
                "license": licence, "license_card": self.license_card,
                "license_confirmed": self.license_confirmed, "gated": e.gated,
                "ram_gb_train": e.ram_gb_train, "ram_source": e.ram_source, "verified": False,
                "trial_run_id": self.trial_run_id}  # fmt: skip


# --------------------------------------------------------------------------------------------- store
def _store(root: Path) -> Path:
    return root / STORE_REL


def save(root: Path, record: UserModelRecord) -> None:
    folder = _store(root)
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (folder.parent, folder)):  # never write through a link out of the data root
        raise OSError("the user asset store must not be a symlink")
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(record.model_dump_json(indent=1))
        Path(tmp).replace(folder / f"{record.entry.id}.json")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load(root: Path, model_id: str) -> UserModelRecord | None:
    """The record, or None when absent or not valid (a hand-edited file never becomes a model)."""
    if len(model_id) > MAX_ID or not USER_ID.fullmatch(model_id):
        return None
    try:
        return UserModelRecord.model_validate(read_small_json(_store(root) / f"{model_id}.json", RECORD_MAX))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, ValidationError):
        logger.warning("user model record %s is not valid: ignored", model_id)
        return None


def picked_user_models(workflow: Mapping[str, Any], node_id: str) -> list[str]:
    """Ids of the user-added models a step reads straight from a model.hf_pick wired into it (MODEL-005)."""
    nodes = {n.get("id"): n for n in workflow.get("nodes", []) if isinstance(n, Mapping)}
    ids: list[str] = []
    for edge in workflow.get("edges", []):
        if not isinstance(edge, Mapping) or (edge.get("to") or [None])[0] != node_id:
            continue
        source = nodes.get((edge.get("from") or [None])[0]) or {}
        params = source.get("params")
        user = (
            params.get("user_model") if isinstance(params, Mapping) and source.get("type") == "model.hf_pick" else None
        )
        if isinstance(user, str) and user and user not in ids:
            ids.append(user)
    return ids


def list_all(root: Path) -> list[UserModelRecord]:
    folder = _store(root)
    names = sorted(p.stem for p in folder.glob("user--*.json")) if folder.is_dir() else []
    return [r for r in (load(root, n) for n in names) if r is not None]


def delete(root: Path, model_id: str) -> bool:
    if len(model_id) > MAX_ID or not USER_ID.fullmatch(model_id):
        return False
    path = _store(root) / f"{model_id}.json"
    existed = path.is_file()
    path.unlink(missing_ok=True)
    (_store(root) / f"{model_id}.train.yaml").unlink(missing_ok=True)
    return existed


def confirm_license(root: Path, model_id: str) -> UserModelRecord | None:
    record = load(root, model_id)
    if record is None:
        return None
    confirmed = record.model_copy(update={"license_confirmed": True, "license_confirmed_at": time.time(),
                                          "license_confirmed_sha": record.entry.revision})  # fmt: skip
    save(root, confirmed)
    return confirmed


def downloadable_entry(root: Path, model_id: str) -> CatalogEntry:
    """The entry whose weights may be fetched: its licence confirmed for exactly its pinned revision (SEC-005)."""
    record = load(root, model_id)
    if record is None:
        from vlm_pipeline.app.downloads import DownloadUnknownError

        raise DownloadUnknownError("not an added model", params={})
    e = record.entry
    if not record.license_confirmed or record.license_confirmed_sha != e.revision:
        raise LicenseNotConfirmedError(
            "licence not confirmed for this revision",
            params={"name": e.display_name, "repo": e.repo, "license": e.license, "catalog_id": e.id, "kind": "model"},
        )
    return e


def save_template(root: Path, record: UserModelRecord) -> str:
    """Write the training template rendered from the record's family; its sha256 (kept in the record)."""
    family = next((f for f in resolver.load_families().families if f.id == record.family), None)
    if family is None:
        raise resolver.ModelArchUnsupportedError("family no longer offered", params={
            "repo": record.entry.repo, "revision": record.entry.revision[:12], "architecture": record.architecture,
            "model_type": "unknown"})  # fmt: skip
    folder = _store(root)
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (folder.parent, folder)):
        raise OSError("the user asset store must not be a symlink")
    data = yaml.safe_dump(resolver.render_template(family), sort_keys=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-", suffix=".yaml")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        Path(tmp).replace(folder / f"{record.entry.id}.train.yaml")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def training_template(root: Path, entry: CatalogEntry) -> dict[str, object]:
    """The training template of any model: the catalog's file, or a user model's after its sha256 check.

    The one way the app reads a template (train, trial pixels/decoding). A user template is read once (no-follow,
    regular file, size cap), hashed and parsed from the same bytes: nothing can swap it between check and use.
    """
    from vlm_pipeline.train.recipe import RecipeError, load_template, template_from_bytes, template_path

    if entry.tier != "user":
        return load_template(template_path(root, entry))
    record = load(root, entry.id)
    path = _store(root) / f"{entry.id}.train.yaml"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise OSError("not a regular file")
            data = fh.read(TEMPLATE_MAX + 1)
    except OSError:
        raise RecipeError(f"no training template for model {entry.id!r}") from None
    if record is None or record.template_sha256 is None or len(data) > TEMPLATE_MAX:
        raise RecipeError(f"no verified training template for model {entry.id!r}")
    if hashlib.sha256(data).hexdigest() != record.template_sha256:
        raise RecipeError("the training template of this model changed on disk: add the model again")
    return template_from_bytes(data, path.name)


LIMITS_REL = Path("outputs") / "system" / "user_limits.json"
MAX_MODEL_GB_RANGE = (1.0, 1000.0)


class UserLimits(BaseModel):
    """What the user may raise in Settings (ADR-021 A1): the size cap of one user-added model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_model_gb: float = Field(DEFAULT_MAX_MODEL_GB, ge=MAX_MODEL_GB_RANGE[0], le=MAX_MODEL_GB_RANGE[1])


def read_limits(root: Path) -> UserLimits:
    """The saved limits, or the defaults (absent, unreadable or invalid: never above the default by accident)."""
    try:
        return UserLimits.model_validate(read_small_json(root / LIMITS_REL, 4096))
    except (OSError, ValueError, ValidationError):
        return UserLimits()


def save_limits(root: Path, limits: UserLimits) -> None:
    path = root / LIMITS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(limits.model_dump_json())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def admin_cap_gb() -> float | None:
    """The admin policy's ``[download] max_gb`` (G-5): Settings may tighten below it, never go above (security)."""
    from vlm_pipeline.common.app_config import read_policy

    policy = read_policy()
    cap = policy.get("download", {}).get("max_gb") if isinstance(policy, dict) else None
    return float(cap) if isinstance(cap, int | float) and not isinstance(cap, bool) and cap > 0 else None


def max_model_gb(root: Path) -> float:
    """The per-model cap for user-added models (ADR-021 A1).

    The default 40 GB or the user's setting, never above the admin policy's download cap.
    """
    cap = admin_cap_gb()
    value = read_limits(root).max_model_gb
    return min(value, cap) if cap is not None else value


# --------------------------------------------------------------------------------------------- resolve (host side)
class _PlannedFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    size: int = Field(ge=0)


class ResolveMeta(BaseModel):
    """``stage/resolve.json`` as the job wrote it: untrusted, so every field is typed and bounded."""

    model_config = ConfigDict(extra="forbid")

    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    gated: bool
    license: str | None
    all_files: list[str] = Field(max_length=2000)
    files: list[_PlannedFile] = Field(max_length=2000)
    total_bytes: int = Field(ge=0)
    unreadable: list[str]
    license_name: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    dataset: dict[str, object] | None = None  # a dataset resolve: validated by app.user_datasets.DatasetBlock

    @model_validator(mode="after")
    def _names(self) -> Self:
        if not all(name_ok(n) for n in self.all_files) or not all(name_ok(f.name) for f in self.files):
            raise ValueError("a file name the download rules refuse")
        if not set(self.unreadable) <= set(meta_caps()):
            raise ValueError("unreadable names a file the job does not read")
        if self.total_bytes != sum(f.size for f in self.files):
            raise ValueError("total_bytes does not add up")
        return self


def read_resolve(run_dir: Path) -> ResolveMeta:
    try:
        return ResolveMeta.model_validate(read_small_json(run_dir / "stage" / "resolve.json", RESOLVE_MAX))
    except (OSError, ValueError, ValidationError) as exc:
        raise resolver.RepoFilesInvalidError(f"resolve.json: {type(exc).__name__}") from None


def finish_model(root: Path, run_dir: Path, repo: str) -> UserModelRecord:
    """The record of a resolved repo (not yet saved), or the catalog error that says why it cannot be added."""
    meta = read_resolve(run_dir)
    if meta.unreadable:  # too big to check is never "absent" (security H2 R1)
        raise resolver.ModelNeedsRemoteCodeError(
            "a config file is too large to check",
            params={"repo": repo, "revision": meta.sha[:12], "reason": "unreadable_config",
                    "file": resolver.clean_name(meta.unreadable[0])},
        )  # fmt: skip
    folder = run_dir / "stage" / "meta"
    caps = meta_caps()
    configs = {
        name: resolver.read_repo_json(folder / name, caps[name])
        for name in resolver.CONFIG_FILES
        if name in meta.all_files
    }
    family, facts = resolver.resolve_family(configs, meta.all_files, repo=repo, revision=meta.sha,
                                            families=resolver.load_families())  # fmt: skip
    weights = [f for f in meta.files if f.name.endswith(".safetensors")]
    if not weights:
        raise resolver.ModelArchUnsupportedError(
            "no safetensors weights",
            params={"repo": repo, "revision": meta.sha[:12], "architecture": facts.architecture,
                    "model_type": facts.model_type, "reason": "no_safetensors"},
        )  # fmt: skip
    params_b = None
    if INDEX_NAME in meta.all_files:
        params_b = resolver.params_b_from_index(resolver.read_repo_json(folder / INDEX_NAME, caps[INDEX_NAME]),
                                                facts.torch_dtype)  # fmt: skip
    if params_b is None:  # one shard and no index: the weights' bytes at the declared dtype (an upper bound)
        dtype_bytes = 4 if facts.torch_dtype == "float32" else 2
        params_b = round(sum(f.size for f in weights) / dtype_bytes / 1e9, 2) or 0.01
    size_gb = meta.total_bytes / GIB
    cap = max_model_gb(root)
    if size_gb > cap:
        raise UserModelTooLargeError(size_gb, cap)
    entry = resolver.user_entry(repo=repo, revision=meta.sha, family=family, params_b=params_b, disk_gb=size_gb,
                                license_id=normalize_license(meta.license, meta.license_name),
                                gated=meta.gated)  # fmt: skip
    return UserModelRecord(entry=entry, family=family.id, architecture=facts.architecture, added_at=time.time(),
                           license_card=_card_text(_card_words(meta.license, meta.license_name)))  # fmt: skip


def _card_words(license_id: str | None, name: str | None) -> str | None:
    """What the card says, for the UI: ``other (qwen)`` when it names its licence in license_name."""
    return f"{license_id} ({name})" if license_id and name else license_id


def _card_text(value: object) -> str | None:
    """The licence words of a card for the UI: plain text, control characters dropped, capped (untrusted)."""
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"[\x00-\x1f\x7f]", "", value).strip()[:200] or None


__all__ = [
    "DEFAULT_MAX_MODEL_GB",
    "LicenseNotConfirmedError",
    "ResolveMeta",
    "UserLimits",
    "UserModelRecord",
    "UserModelTooLargeError",
    "admin_cap_gb",
    "confirm_license",
    "delete",
    "downloadable_entry",
    "finish_model",
    "list_all",
    "load",
    "max_model_gb",
    "read_limits",
    "read_resolve",
    "save",
    "save_limits",
    "save_template",
    "training_template",
]
