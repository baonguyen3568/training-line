"""What a container job may run, per kind (ADR-015 runner protocol): shared by the app launcher and vlm-runner."""

from __future__ import annotations

# The first argv items a kind may run; the job's own arguments follow. One place to read what jobs exist.
ALLOWED_ARGV: dict[str, tuple[tuple[str, ...], ...]] = {
    "train": (("automodel",), ("python", "-m", "vlm_pipeline.train.merge_lora")),
    "eval": (("python", "/workspace/scripts/qa/eval_cord_worker.py"),),
    "quantize": (("python", "-m", "vlm_pipeline.quantize.ptq"),),
    "serve": (("python", "-m", "vlm_pipeline.serve.trial"), ("python", "-m", "vlm_pipeline.serve.bench")),
    # No first-party module yet (downloads are manual, D1): every argv is refused until one exists and is reviewed.
    "probe": (),
    "download": (),
}


def job_kind(argv: tuple[str, ...]) -> str | None:
    """The kind whose allowed prefix ``argv`` starts with (token-wise), or None."""
    for kind, prefixes in ALLOWED_ARGV.items():
        if any(argv[: len(p)] == p for p in prefixes):
            return kind
    return None


__all__ = ["ALLOWED_ARGV", "job_kind"]
