"""Token leak gate (scripts/security/scan_tokens.py). Fake tokens are built at runtime so this file never matches."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "security" / "scan_tokens.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("scan_tokens", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


st = _load()
FAKE_HF = "hf" + "_" + "AbCdEfGhIj1234567890KlMnOpQrStUvWx"  # 34 chars after the prefix, like real tokens
FAKE_HF_LETTERS = "hf" + "_" + "AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"  # current HF format: 34 letters, no digit
FAKE_NGC = "nvapi" + "-" + "x" * 64


@pytest.mark.parametrize(
    "text",
    [
        f"HF_TOKEN={FAKE_HF}",
        f"HF_TOKEN={FAKE_HF_LETTERS}",
        f'headers = {{"Authorization": "Bearer {FAKE_HF}"}}',
        f"log line ... token={FAKE_HF} ...",
        f"NGC_API_KEY={FAKE_NGC}",
        "WANDB_API_KEY=" + "0123456789abcdef" * 2 + "01234567",
    ],
)
def test_finds_real_shaped_credentials(text: str) -> None:
    leaks = list(st.scan_text("f.txt", text))
    assert leaks
    assert all(
        f not in str(leak) for f in (FAKE_HF, FAKE_HF_LETTERS, FAKE_NGC) for leak in leaks
    )  # never printed in full


@pytest.mark.parametrize(
    "text",
    [
        "from huggingface_hub import hf_hub_download",
        "encoder.auto_map: c-radiov2-h--hf_model.RADIOConfig",
        "hf_nemotron_parse_config.py hf_nemotron_parse_modeling.py",
        "HF_TOKEN_PATH=/run/secrets/hf_token",
        "hf_" + "x" * 34,  # one case only: an identifier, not a token
        "hf_" + "X1" * 17,
        "my_hf_" + "AbCdEfGhIj1234567890KlMnOpQrStUvWx",  # part of a longer identifier
    ],
)
def test_ignores_code_that_merely_starts_with_hf(text: str) -> None:
    assert list(st.scan_text("f.py", text)) == []


def test_paths_mode_and_skips(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text(f"ok\nleak {FAKE_HF}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text(FAKE_HF, encoding="utf-8")
    (tmp_path / "link.log").symlink_to(tmp_path / "logs" / "run.log")
    leaks = st.scan_files(st.walk([tmp_path]), base=tmp_path)
    assert [(leak.path, leak.line, leak.kind) for leak in leaks] == [("logs/run.log", 2, "HF_TOKEN")]


def test_tracked_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)  # noqa: S603,S607
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text(FAKE_HF, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)  # noqa: S603,S607
    monkeypatch.chdir(tmp_path)
    assert st.main(["tracked"]) == 0  # untracked files are not the CI gate's business
    (tmp_path / "a.py").write_text(f"T = '{FAKE_HF}'\n", encoding="utf-8")
    assert st.main(["tracked"]) == 1


def test_usage() -> None:
    assert st.main([]) == 2
    assert st.main(["paths"]) == 2
