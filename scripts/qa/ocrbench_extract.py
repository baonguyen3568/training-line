"""Shim: the extraction lives in ``vlm_pipeline.eval.ocrbench_extract`` (narrow jobs mount src/, not scripts/)."""

from vlm_pipeline.eval.ocrbench_extract import main

if __name__ == "__main__":
    raise SystemExit(main())
