"""common.run_log: the host reads a job's log only as a regular file reached without links (bao-security R2/B1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vlm_pipeline.common.run_log import RunLogError, open_run_log, tail_run_log


def test_tail_of_a_regular_log(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stdout.log").write_text("".join(f"{i}\n" for i in range(10)))
    assert tail_run_log(tmp_path, 3) == ["7", "8", "9"]


def test_links_fifos_and_bad_names_are_refused(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    secret = tmp_path / "secret"
    secret.write_text("x")
    (tmp_path / "logs" / "stdout.log").symlink_to(secret)
    os.mkfifo(tmp_path / "logs" / "fifo.log")
    for name in ("stdout.log", "fifo.log", "../secret", "missing.log"):
        with pytest.raises(RunLogError):
            open_run_log(tmp_path, name=name)
