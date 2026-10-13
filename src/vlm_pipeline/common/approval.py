"""The security sign-off mark shared by our lock files (IMAGES.lock, LICENSES.lock). Stdlib only."""

from __future__ import annotations

import re

APPROVED_MARK = "bao-security APPROVED"
_APPROVED_RE = re.compile(re.escape(APPROVED_MARK) + r"(?:$|[\s(])")  # "APPROVED-PENDING" is not approved


def is_security_approved(verdict: str) -> bool:
    """A lock row's review column must START with exactly ``bao-security APPROVED`` (no free-text parsing)."""
    return _APPROVED_RE.match(verdict) is not None


__all__ = ["APPROVED_MARK", "is_security_approved"]
