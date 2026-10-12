"""Pre-run check before taking the GPU lock (ADR-005 D8). Thin wrapper over ``common.preflight.preflight``.

Prints the verdict as JSON on stdout and exits with ``ExitCode`` (0 ok, 2 memory/swap, 3 disk, 4 lock busy).
Read-only: never drops caches, never touches a lock.

    uv run python scripts/cleanup/preflight.py --est-peak-gb 16 --ckpt-gb 1
"""

from __future__ import annotations

import argparse
import sys

from vlm_pipeline.common.preflight import preflight

# ADR-005 D8: a job that does not declare its peak is assumed to need 40 GB.
DEFAULT_EST_PEAK_GB = 40.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--est-peak-gb", type=float, default=DEFAULT_EST_PEAK_GB, help="expected peak unified memory")
    parser.add_argument("--ckpt-gb", type=float, default=0.0, help="size of one checkpoint the job will write")
    args = parser.parse_args(argv)
    if args.est_peak_gb <= 0 or args.ckpt_gb < 0:
        parser.error("--est-peak-gb must be > 0 and --ckpt-gb >= 0")

    verdict = preflight(args.est_peak_gb, ckpt_gb=args.ckpt_gb)
    print(verdict.to_json())
    for reason in verdict.reasons:
        print(f"preflight: {reason}", file=sys.stderr)
    return int(verdict.exit_code)


if __name__ == "__main__":
    sys.exit(main())
