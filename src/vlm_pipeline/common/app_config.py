"""Infrastructure configuration (ADR-015): app id, admin policy path, and ``.env.local`` - startup settings only.

``.env.local`` (repo root, gitignored, 0600) is the single source of INFRASTRUCTURE settings (port, data root,
profile, runner socket...). User-editable settings live in the app DB (D-7), admin policy in ``policy.toml``.
The file also holds secrets for compose (``secrets:``): this reader returns ONLY the allowlisted infra keys, so a
token in the file never enters the app process through it. Process environment wins over the file (compose passes
the same values), and both lose to nothing else: no other config file is read.
"""

from __future__ import annotations

import logging
import os
import pwd
import stat
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
Profile = Literal["personal", "shared_dev"]
APP_ID = "ntc-training"  # the one name for config dirs, compose project, labels (user decision: "NTC-Training")
ENV_FILE = ".env.local"
POLICY_REL = Path(".config") / APP_ID / "policy.toml"  # admin-owned, read-only for the app (security 7.2)
INFRA_KEYS = frozenset(
    {"VLM_PORT", "VLM_DATA_ROOT", "VLM_PROFILE", "VLM_RUNNER_SOCKET", "VLM_RETENTION_SCHEDULE"}
)  # never a secret: tokens stay out of this reader by construction


class InfraConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    port: int = Field(7870, ge=1024, le=65535)
    data_root: Path | None = None  # None: the project dir (.cache/, outputs/, data/)
    profile: Profile | None = None  # effective: admin policy first, env/.env.local may only tighten it
    runner_socket: Path | None = None  # set in compose: jobs go through vlm-runner, never a local docker CLI
    retention_schedule: Literal["off", "dry_run", "apply"] | None = None


def _parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key in INFRA_KEYS:
            values[key] = value
    return values


def read_env_local(root: Path) -> dict[str, str]:
    """Allowlisted keys of ``<root>/.env.local`` (missing file -> {}). A symlink or a non-regular file is ignored."""
    path = root / ENV_FILE
    try:
        st = path.lstat()
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(st.st_mode):
        return {}
    return _parse(path.read_text(encoding="utf-8", errors="replace"))


_FORBIDDEN_DATA_ROOTS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/boot", "/proc", "/sys", "/dev", "/var/run", "/run")


def _home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def check_data_root(value: str) -> Path:
    """VLM_DATA_ROOT: every delete path (retention, cache) is confined to it, so it must be a narrow, owned folder.

    Absolute; not ``/``, not the user's home or any of its ancestors, not under system dirs; when it exists: a real
    directory (no symlink) owned by this user and not group/other-writable.
    """
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError("VLM_DATA_ROOT must be an absolute path")
    if raw.is_symlink():
        raise ValueError("VLM_DATA_ROOT must not be a symlink")
    path = raw.resolve()
    home = _home()
    if path == Path("/") or path == home or path in home.parents:
        raise ValueError("VLM_DATA_ROOT must be a dedicated folder, not / or your home (or above it)")
    if any(path == Path(d) or path.is_relative_to(d) for d in _FORBIDDEN_DATA_ROOTS):
        raise ValueError("VLM_DATA_ROOT must not be inside a system directory")
    if path.exists():
        st = path.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
            raise ValueError("VLM_DATA_ROOT must be a directory you own, not writable by group/others")
    return path


def policy_path() -> Path:
    """``~/.config/ntc-training/policy.toml`` of the process user, from passwd and never from ``$HOME``.

    The environment must not be able to point the app at a looser policy.
    """
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / POLICY_REL


def _unsafe(st: os.stat_result) -> bool:
    return bool(st.st_mode & 0o022) or st.st_uid not in (0, os.getuid())


def read_policy(path: Path | None = None) -> dict[str, Any] | None:
    """The admin policy file parsed, or None when absent or unusable (logged). THE one reader of policy.toml.

    Fails closed on: a symlink (O_NOFOLLOW), a file or parent dir writable by group/others or owned by someone other
    than this user or root, a non-regular file, invalid TOML.
    """
    path = path if path is not None else policy_path()
    try:
        if _unsafe(path.parent.stat()):  # others could swap the file by rename
            logger.warning("policy dir %s is group/other-writable or foreign-owned: ignored", path.parent)
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP for a symlink
        logger.warning("policy file %s unusable (%s): ignored", path, exc)
        return None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if _unsafe(st) or not stat.S_ISREG(st.st_mode):
            logger.warning("policy file %s is not a regular file owned by this user/root and 0644: ignored", path)
            return None
        try:
            return tomllib.load(fh)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            logger.warning("policy file %s invalid (%s): ignored", path, exc)
            return None


def effective_profile(local: str | None, policy: dict[str, Any] | None) -> Profile | None:
    """Admin ``[deployment] profile`` is the base; env/.env.local may only TIGHTEN it (personal -> shared_dev).

    Without an admin policy (a personal machine) the local value decides.
    """
    if local not in (None, "personal", "shared_dev"):
        raise ValueError(f"VLM_PROFILE must be personal|shared_dev, got {local!r}")
    deployment = policy.get("deployment") if policy else None
    admin = deployment.get("profile") if isinstance(deployment, dict) else None
    if admin is None:
        return local  # type: ignore[return-value]  # validated above
    if admin not in ("personal", "shared_dev"):  # fail closed on a value nobody can read
        logger.warning("deployment.profile must be personal|shared_dev, got %r: treated as shared_dev", admin)
        return "shared_dev"
    if admin == "shared_dev" or local == "shared_dev":
        return "shared_dev"
    return "personal"


def load_infra(
    root: Path, environ: dict[str, str] | None = None, *, policy: dict[str, Any] | None = None
) -> InfraConfig:
    """``policy`` defaults to the admin file (``read_policy()``); pass ``{}`` for none."""
    env = os.environ if environ is None else environ
    merged = {**read_env_local(root), **{k: v for k, v in env.items() if k in INFRA_KEYS}}
    data_root = merged.get("VLM_DATA_ROOT")
    return InfraConfig(
        port=int(merged.get("VLM_PORT", 7870)),
        data_root=check_data_root(data_root) if data_root else None,
        profile=effective_profile(merged.get("VLM_PROFILE") or None, read_policy() if policy is None else policy),
        runner_socket=Path(merged["VLM_RUNNER_SOCKET"]) if merged.get("VLM_RUNNER_SOCKET") else None,
        retention_schedule=merged.get("VLM_RETENTION_SCHEDULE") or None,  # the model validates
    )


def data_root(root: Path, config: InfraConfig | None = None) -> Path:
    """Where .cache/, outputs/ and data/ live: VLM_DATA_ROOT (checked), else the project dir."""
    cfg = config or load_infra(root)
    return (cfg.data_root or root).resolve()


__all__ = [
    "APP_ID",
    "ENV_FILE",
    "INFRA_KEYS",
    "POLICY_REL",
    "InfraConfig",
    "check_data_root",
    "data_root",
    "effective_profile",
    "load_infra",
    "policy_path",
    "read_env_local",
    "read_policy",
]
