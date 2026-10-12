"""Spec: a finished run must carry the fields ADR-002 B2 requires to call a result reproducible.

Found on the first real runner artifact (outputs/runs/20261012-1653-p1-parse-v1.1-c557/run.json): ``git_commit`` and
``container_digest`` were null, so the dashboard would show "not reproducible" for every run.
ASSUMED: ``vlm_pipeline.common.runinfo.reproducibility_gaps(manifest) -> list[str]`` (names of missing fields;
empty list = reproducible) - agree the name with bao-backend.
"""

from __future__ import annotations

from typing import Any

import pytest

from vlm_pipeline.common.runinfo import RunManifest, RunStatus

COMPLETE: dict[str, Any] = {
    "run_id": "r1",
    "name": "smoke",
    "stage": "train",
    "task": "lora",
    "status": RunStatus.COMPLETED,
    "git_commit": "a" * 40,
    "git_dirty": False,
    "container_digest": "sha256:" + "b" * 64,
    "config_hash": "c" * 64,
    "seed": 1234,
    "base_model": "nvidia/NVIDIA-Nemotron-Parse-v1.1",
    "base_model_revision": "fd1e3c6ba0b61a6dcab5a905f0ff92c5b19f14a4",
}


def _gaps(**overrides: Any) -> list[str]:
    from vlm_pipeline.common.runinfo import reproducibility_gaps

    return reproducibility_gaps(RunManifest(**{**COMPLETE, **overrides}))


def test_complete_manifest_has_no_gaps() -> None:
    assert _gaps() == []


@pytest.mark.parametrize("field", ["git_commit", "container_digest", "config_hash", "seed", "base_model_revision"])
def test_each_required_field_is_reported(field: str) -> None:
    assert field in _gaps(**{field: None})


def test_dirty_tree_is_a_gap() -> None:
    """A run from uncommitted code cannot be reproduced from ``git_commit`` alone."""
    assert "git_dirty" in _gaps(git_dirty=True)


def test_short_or_branch_commit_is_a_gap() -> None:
    assert "git_commit" in _gaps(git_commit="main")
    assert "git_commit" in _gaps(git_commit="a" * 7)
