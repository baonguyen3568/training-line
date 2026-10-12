"""Gate (bao-security, 2026-10-12): Python libraries in the worker image must not open network connections once the
"no phone home" env block is set (NO_ALBUMENTATIONS_UPDATE, HF_HUB_OFFLINE, HF_HUB_DISABLE_TELEMETRY, DO_NOT_TRACK,
WANDB_MODE=offline).

Method: run the action in a fresh interpreter whose socket.getaddrinfo / connect / connect_ex are replaced by
counters that refuse. DNS is counted because under --network none it is the ONLY step that happens (connect is never
reached). Every case has a control run without the variable that must show >= 1 lookup, so a gate can never pass
because the probe saw nothing. Timing is not used: with --network none DNS fails instantly either way.
Limitation: wandb >= 0.18 does its networking in the Go ``wandb-core`` process, invisible to this probe; for wandb we
only assert the run really is offline (the container's --network none is the actual barrier).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("huggingface_hub")

pytestmark = pytest.mark.integration

PROBE_HEADER = """
import socket
calls = []
def _refuse(*args, **kwargs):
    calls.append(repr(args[:2]))
    raise OSError("network disabled by test")
socket.getaddrinfo = _refuse
socket.socket.connect = lambda self, address: _refuse(address)
socket.socket.connect_ex = lambda self, address: calls.append(repr(address)) or 1
def report():
    print("LOOKUPS", len(calls), calls[:3])
try:
"""

PARSE = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
CASES = {
    # name: (action, env that must silence it)
    "albumentations": ("import albumentations, cv2", {"NO_ALBUMENTATIONS_UPDATE": "1"}),
    "huggingface_hub": (
        f"import huggingface_hub as h\n    h.model_info({PARSE!r})",
        {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
    ),
    "transformers": (
        f"from transformers import AutoConfig\n    AutoConfig.from_pretrained({PARSE!r})",
        {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
    ),
    "datasets": (
        "from datasets import load_dataset\n    load_dataset('naver-clova-ix/cord-v2', split='test[:1]')",
        {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
    ),
}
SILENCERS = {
    "NO_ALBUMENTATIONS_UPDATE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
}
# Control runs must be online even when the image defaults to offline (26.08-5 sets the *_OFFLINE vars in ENV).
FORCE_ONLINE = {"HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}


def _lookups(action: str, env: dict[str, str]) -> tuple[int, str]:
    code = PROBE_HEADER + f"    {action}\nexcept Exception:\n    pass\nreport()\n"
    out = subprocess.run(  # noqa: S603 - fixed interpreter + constant probe code
        [sys.executable, "-c", code],
        env=env,  # HOME comes from the caller (container sets it)
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    text = out.stdout + out.stderr
    line = next((ln for ln in text.splitlines() if ln.startswith("LOOKUPS ")), "")
    if not line:
        raise AssertionError(f"probe printed no result:\n{text[-2000:]}")
    return int(line.split()[1]), line


def _without_silencers() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in SILENCERS}


@pytest.mark.parametrize("name", sorted(CASES))
def test_control_probe_sees_the_network_attempt(name: str) -> None:
    """Without the variables the action does try the network: otherwise the gate below would be vacuous."""
    action, _ = CASES[name]
    count, line = _lookups(action, {**_without_silencers(), **FORCE_ONLINE})
    assert count >= 1, f"{name}: control saw no lookup, the probe cannot prove anything ({line})"


@pytest.mark.parametrize("name", sorted(CASES))
def test_variables_silence_the_library(name: str) -> None:
    action, env = CASES[name]
    count, line = _lookups(action, {**_without_silencers(), **env})
    assert count == 0, f"{name}: still looks up the network with {env} ({line}): report to bao-security"


def _env_has(keys: dict[str, str]) -> bool:
    return all(os.environ.get(k, "").lower() in {"1", "true"} for k in keys)


@pytest.mark.parametrize(
    "name",
    [
        # Each library only xfails while ITS variables are missing from the environment the test runs in. Image ENV
        # (26.08-4) sets NO_ALBUMENTATIONS_UPDATE / HF_HUB_DISABLE_TELEMETRY / DO_NOT_TRACK / WANDB_MODE; HF_HUB_OFFLINE
        # is set by the job runner at run time, so the HF cases pass when run the way jobs run.
        pytest.param(
            n, marks=pytest.mark.xfail(not _env_has(CASES[n][1]), strict=True, reason=f"env lacks {CASES[n][1]}")
        )
        for n in sorted(CASES)
    ],
)
def test_runtime_env_silences_the_library(name: str) -> None:
    action, _ = CASES[name]
    count, line = _lookups(action, dict(os.environ))
    assert count == 0, line


def test_wandb_offline_mode_is_really_offline(tmp_path: object) -> None:
    pytest.importorskip("wandb")
    code = (
        "import os, wandb\n"
        "run = wandb.init(mode=os.environ['WANDB_MODE'], project='gate', dir=os.environ['WANDB_DIR'])\n"
        "print('OFFLINE', run.settings._offline, 'NOOP', run.settings._noop)\n"
        "run.finish()\n"
    )
    env = {**os.environ, "WANDB_MODE": "offline", "WANDB_DIR": str(tmp_path), "WANDB_SILENT": "true"}
    env.pop("WANDB_API_KEY", None)
    out = subprocess.run(  # noqa: S603 - fixed interpreter + constant code
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180, check=True
    )
    assert "OFFLINE True" in out.stdout, out.stdout + out.stderr[-1000:]
