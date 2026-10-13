"""LocalDockerLauncher (ADR-015 JobLauncher): the container runs the image id the gate approved, not the tag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common import launcher as launcher_mod
from vlm_pipeline.common.jobrun import ContainerJob


class _Proc:
    def wait(self) -> int:
        return 0


def test_run_uses_the_gated_image_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kw: Any) -> _Proc:
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(launcher_mod.subprocess, "Popen", fake_popen)
    job = ContainerJob(name="r1", image="vlm-automodel:26.08-6", argv=("true",), uid=1000, gid=1000)
    pinned = "sha256:" + "ab" * 32
    run = launcher_mod.LocalDockerLauncher().run
    rc = run(job, tmp_path, lock_token=None, log_path=tmp_path / "l", image_id=pinned)
    assert rc == 0
    argv = seen["argv"]
    assert pinned in argv
    assert "vlm-automodel:26.08-6" not in argv  # the tag is never re-resolved between the gate and docker run
    assert argv[argv.index(pinned) + 1] == "true"
