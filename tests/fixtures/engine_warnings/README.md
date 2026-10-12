# Engine warning samples

One real log line per `jobrun.ENGINE_WARNINGS` code (`<code>.log`), copied verbatim from a job log.
Re-capture them when the training image changes; `test_engine_warning_fixtures_match` fails if a code has no sample.

| code | image | source run |
|---|---|---|
| bf16_adam_params | vlm-automodel:26.08-5 (NeMo AutoModel 26.08) | outputs/runs/20261012-1943-p3-parse-template-69db |
