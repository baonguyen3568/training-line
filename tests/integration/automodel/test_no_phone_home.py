"""Gate (bao-security, 2026-10-12): importing the Parse processor's dependencies must not open network connections.

albumentations 2.0.8 checks PyPI for updates on import unless NO_ALBUMENTATIONS_UPDATE is "1"/"true"; the image
and runner set it with the other "no phone home" env vars. The check counts socket.connect calls in a fresh
interpreter (and DNS lookups, which come first and are all that happens under --network none), so it holds with
or without a network (timing would not: --network none fails DNS instantly).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("albumentations")

pytestmark = pytest.mark.integration

PROBE = """
import socket
calls = []
_orig = socket.socket.connect
def _connect(self, address):
    calls.append(repr(address))
    raise OSError("network disabled by test")
socket.socket.connect = _connect
socket.socket.connect_ex = lambda self, address: calls.append(repr(address)) or 1
_orig_gai = socket.getaddrinfo
def _getaddrinfo(host, *args, **kwargs):  # DNS comes first: with --network none it fails before any connect()
    calls.append(repr(host))
    raise OSError("network disabled by test")
socket.getaddrinfo = _getaddrinfo
import albumentations  # noqa: F401
import cv2  # noqa: F401
print("CONNECT_CALLS", len(calls), calls[:3])
"""

NO_PHONE_HOME = os.environ.get("NO_ALBUMENTATIONS_UPDATE", "").lower() in {"1", "true"}


def _run(env: dict[str, str]) -> str:
    out = subprocess.run(  # noqa: S603 - fixed interpreter + constant probe code
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=120, check=True
    )
    return out.stdout + out.stderr


@pytest.mark.xfail(not NO_PHONE_HOME, strict=True, reason="image env lacks NO_ALBUMENTATIONS_UPDATE (backend)")
def test_image_env_imports_without_network() -> None:
    output = _run(dict(os.environ))
    assert "CONNECT_CALLS 0 " in output, output
    assert "Error fetching version info" not in output


def test_env_var_is_what_disables_the_check() -> None:
    """Control: the same import WITHOUT the variable does try to connect (so the gate above is not vacuous)."""
    env = {k: v for k, v in os.environ.items() if k != "NO_ALBUMENTATIONS_UPDATE"}
    with_check = _run(env)
    without_check = _run({**env, "NO_ALBUMENTATIONS_UPDATE": "1"})
    assert "CONNECT_CALLS 0 " not in with_check, with_check
    assert "CONNECT_CALLS 0 " in without_check, without_check
