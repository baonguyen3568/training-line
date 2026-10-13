"""Vendored licence texts: ``configs/licenses/<id>.txt`` locked by ``LICENSES.lock`` (Proposal 012 §2.1).

Single reader of the lock for the catalog tests and the export. Fail-closed: a text counts only when its row carries
a security APPROVED verdict and the file's sha256 matches the row. Stdlib only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from vlm_pipeline.common.approval import is_security_approved

LICENSES_DIR = Path("configs") / "licenses"
LICENSES_LOCK = LICENSES_DIR / "LICENSES.lock"


@dataclass(frozen=True)
class LockedLicense:
    id: str
    sha256: str
    approved: bool


def read_lock(root: Path) -> dict[str, LockedLicense]:
    """Rows of ``LICENSES.lock``: ``id | url | sha256 of the .txt | ... | security verdict`` (verdict = last column).

    Approved = the verdict starts with ``bao-security APPROVED`` (same mark as IMAGES.lock).
    """
    path = root / LICENSES_LOCK
    if not path.is_file():
        return {}
    rows: dict[str, LockedLicense] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 4:
            rows[cols[0]] = LockedLicense(id=cols[0], sha256=cols[2], approved=is_security_approved(cols[-1]))
    return rows


def license_text(root: Path, license_id: str) -> Path | None:
    """Verbatim text of ``license_id`` if it is vendored, approved AND matches its locked sha256."""
    row = read_lock(root).get(license_id)
    path = root / LICENSES_DIR / f"{license_id}.txt"
    if row is None or not row.approved or not path.is_file():
        return None
    return path if hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256 else None


__all__ = ["LICENSES_DIR", "LICENSES_LOCK", "LockedLicense", "license_text", "read_lock"]
