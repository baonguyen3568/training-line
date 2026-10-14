"""ADR-015 job kinds: every allowed argv prefix runs with an entrypoint from a closed set (bao-security)."""

from __future__ import annotations

import pytest

from vlm_pipeline.common.job_kinds import (
    ALLOWED_ARGV,
    ENTRYPOINTS,
    NGC_ENTRYPOINT,
    argv_allowed,
    job_entrypoint,
    job_kind,
)


def test_ngc_jobs_keep_nvidias_wrapper_with_the_full_argv() -> None:
    argv = ("automodel", "finetune", "-c", "/workspace/x.yaml")
    assert job_entrypoint("train", argv) == (NGC_ENTRYPOINT, argv)  # wrapper: driver/compat checks, then exec "$@"


def test_the_vllm_batch_runs_python3_itself_not_the_images_vllm_serve() -> None:
    argv = ("python3", "-m", "vlm_pipeline.serve.vllm_batch", "/workspace/outputs/runs/r/batch.json")
    assert job_kind(argv) == "eval"
    assert job_entrypoint("eval", argv) == ("python3", argv[1:])


@pytest.mark.parametrize(
    ("kind", "argv"),
    [
        ("train", ("python", "-m", "vlm_pipeline.quantize.ptq")),  # allowed, but for another kind
        ("eval", ("python3", "-c", "import os")),
        ("eval", ("bash", "-c", "id")),
        ("probe", ("python", "-m", "anything")),
    ],
)
def test_argv_outside_the_kind_is_refused(kind: str, argv: tuple[str, ...]) -> None:
    assert not argv_allowed(kind, argv)
    with pytest.raises(ValueError, match="argv_not_allowed"):
        job_entrypoint(kind, argv)


def test_entrypoints_come_from_the_closed_set_only() -> None:
    assert {ep for rules in ALLOWED_ARGV.values() for _, ep in rules} <= ENTRYPOINTS
    assert len(ENTRYPOINTS) == 2


def test_every_extraction_job_the_app_starts_is_a_data_job() -> None:
    """The installed stack runs jobs through vlm-runner, which refuses an argv no kind allows (G-5 extraction)."""
    from vlm_pipeline.common.job_kinds import job_kind
    from vlm_pipeline.download.extracted import EXTRACTED

    for spec in EXTRACTED.values():
        argv = ("python", "-m", spec.module, "--parquet", "/workspace/x.parquet", "--out", "/workspace/o")
        assert job_kind(argv) == "data"  # never "download": no network for work on files
