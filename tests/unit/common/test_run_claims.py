from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.common.run_claims import (
    CLAIMS_REL,
    ClaimStoreError,
    RunDirClaimedError,
    claim_run,
    is_claimed,
)

RUN = "20261013-1200-p3b-full-parse-c4f6"
NODE = "t-f5a500ced6dc0d43"


def test_a_run_is_claimed_once(tmp_path: Path) -> None:
    path = claim_run(tmp_path, RUN, by="runner")
    assert json.loads(path.read_text())["run_id"] == RUN
    assert path.stat().st_mode & 0o777 == 0o600
    assert is_claimed(tmp_path, RUN)
    with pytest.raises(RunDirClaimedError):
        claim_run(tmp_path, RUN, by="runner")
    claim_run(tmp_path, NODE, by="local")  # graph node run dirs too


def test_a_job_emptying_its_run_dir_cannot_reopen_it(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / RUN
    run_dir.mkdir(parents=True)
    claim_run(tmp_path, RUN, by="runner")
    for child in run_dir.iterdir():  # all a job can touch is its own run dir: the claim lives elsewhere
        child.unlink()
    with pytest.raises(RunDirClaimedError):
        claim_run(tmp_path, RUN, by="runner")


def test_a_planted_symlink_claim_is_refused_and_not_followed(tmp_path: Path) -> None:
    claims = tmp_path / CLAIMS_REL
    claims.mkdir(parents=True)
    target = tmp_path / "victim"
    (claims / RUN).symlink_to(target)
    with pytest.raises(RunDirClaimedError):
        claim_run(tmp_path, RUN, by="runner")
    assert not target.exists()


def test_a_symlinked_claims_folder_is_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "outputs" / "system").mkdir(parents=True)
    (tmp_path / CLAIMS_REL).symlink_to(elsewhere)
    with pytest.raises(ClaimStoreError):
        claim_run(tmp_path, RUN, by="runner")
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("bad", ["../x", "kafka", "20261013-1200-UPPER", "a/b-0123456789abcdef", ""])
def test_bad_run_ids_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="bad_run_id"):
        claim_run(tmp_path, bad, by="runner")
    assert not is_claimed(tmp_path, bad)


def test_nothing_is_created_through_a_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "data"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs" / "system").symlink_to(outside)
    with pytest.raises(ClaimStoreError):
        claim_run(root, RUN, by="runner")
    assert list(outside.iterdir()) == []  # no runner_claims folder made at the symlink's target
