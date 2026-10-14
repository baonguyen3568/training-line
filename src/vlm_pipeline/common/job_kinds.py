"""What a container job may run, per kind (ADR-015 runner protocol): shared by the app launcher and vlm-runner.

Every allowed argv prefix carries the ENTRYPOINT the container runs it with, from a closed set (bao-security):
the image's own ENTRYPOINT never decides, and neither does a request or a config.
- ``NGC_ENTRYPOINT``: NVIDIA's wrapper in NGC-based images. It runs the driver / sm / forward-compat / shm checks of
  ``/opt/nvidia/entrypoint.d`` (the diagnostics D-4 and sm_121 bugs need in the job log), then ``exec "$@"`` the full
  argv, so the program run is still the allowlisted ``argv[0]``.
- ``ARGV0``: ``argv[0]`` itself is the entrypoint, the rest its arguments (vLLM image: ENTRYPOINT is ``vllm serve``
  and only ``python3`` exists).
"""

from __future__ import annotations

from typing import Final

NGC_ENTRYPOINT: Final = "/opt/nvidia/nvidia_entrypoint.sh"
ARGV0: Final = "argv0"
ENTRYPOINTS: Final = frozenset({NGC_ENTRYPOINT, ARGV0})

# (first argv items, entrypoint) a kind may run; the job's own arguments follow. One place to read what jobs exist.
ALLOWED_ARGV: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "train": (
        (("automodel",), NGC_ENTRYPOINT),
        (("python", "-m", "vlm_pipeline.train.merge_lora"), NGC_ENTRYPOINT),
    ),
    "eval": (
        (("python", "/workspace/scripts/qa/eval_cord_worker.py"), NGC_ENTRYPOINT),
        (("python3", "-m", "vlm_pipeline.serve.vllm_batch"), ARGV0),  # vlm-vllm image (B7-3)
    ),
    "quantize": (
        (("python", "-m", "vlm_pipeline.quantize.ptq"), NGC_ENTRYPOINT),
        (("python", "-m", "vlm_pipeline.quantize.qat"), NGC_ENTRYPOINT),  # ADR-022 #10, train image
    ),
    "serve": (
        (("python", "-m", "vlm_pipeline.serve.trial"), NGC_ENTRYPOINT),
        (("python", "-m", "vlm_pipeline.serve.bench"), NGC_ENTRYPOINT),
    ),
    # No first-party module yet: every argv is refused until one exists and is reviewed.
    "probe": (),
    # P018 / security download-egress: the one job with a network (HF-only proxy) and a token file.
    "download": ((("python", "-m", "vlm_pipeline.download.hub"), NGC_ENTRYPOINT),),
    # CPU work on downloaded files, no network, no GPU (the runner refuses gpu): eval data extraction (G-5). Without
    # this kind the extraction argv matched nothing, and the installed stack refused it (argv_not_allowed).
    "data": ((("python", "-m", "vlm_pipeline.eval.ocrbench_extract"), NGC_ENTRYPOINT),),
}
if not all(ep in ENTRYPOINTS for rules in ALLOWED_ARGV.values() for _, ep in rules):  # survives python -O
    raise RuntimeError("ALLOWED_ARGV uses an entrypoint outside the closed set")


def _rule(argv: tuple[str, ...], kind: str | None = None) -> tuple[str, tuple[str, ...], str] | None:
    for k, rules in ALLOWED_ARGV.items():
        if kind is not None and k != kind:
            continue
        for prefix, entrypoint in rules:
            if argv[: len(prefix)] == prefix:
                return k, prefix, entrypoint
    return None


def job_kind(argv: tuple[str, ...]) -> str | None:
    """The kind whose allowed prefix ``argv`` starts with (token-wise), or None."""
    rule = _rule(argv)
    return None if rule is None else rule[0]


def argv_allowed(kind: str, argv: tuple[str, ...]) -> bool:
    """Whether ``argv`` starts with one of ``kind``'s allowed prefixes."""
    return _rule(argv, kind) is not None


def job_entrypoint(kind: str, argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """(``--entrypoint`` value, container command) for an allowed ``argv``; ValueError if it is not allowed.

    NGC wrapper: the whole argv goes to the wrapper. ARGV0: ``argv[0]`` is the entrypoint, ``argv[1:]`` the command.
    """
    rule = _rule(argv, kind)
    if rule is None:
        raise ValueError("argv_not_allowed")
    entrypoint = rule[2]
    return (argv[0], argv[1:]) if entrypoint == ARGV0 else (entrypoint, argv)


__all__ = ["ALLOWED_ARGV", "ARGV0", "ENTRYPOINTS", "NGC_ENTRYPOINT", "argv_allowed", "job_entrypoint", "job_kind"]
