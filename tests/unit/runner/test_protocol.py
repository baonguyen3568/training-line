"""P015 §6.1: the runner refuses anything outside its closed request language, naming the rule, never the input."""

from __future__ import annotations

import json
from typing import Any

import pytest

from vlm_pipeline.common.runner_protocol import MAX_REQUEST_BYTES, ProtocolError, RunRef, Submit, parse_request

RUN = "20261013-1200-p3b-full-parse-c4f6"
GOOD: dict[str, Any] = {
    "command": "submit",
    "run_id": RUN,
    "job_kind": "train",
    "image_tag": "vlm-automodel:26.08-6",
    "argv": ["automodel", f"/workspace/outputs/runs/{RUN}/config.yaml", "--nproc-per-node", "1"],
    "gpu": True,
    "memory_gb": 48,
    "lock_token": "a" * 32,
    "run_dir": f"/app/outputs/runs/{RUN}",
    "inputs": ["/app/.cache/hf/hub/models--nvidia--x", "/app/data/uploads/u1"],
}


def _req(**changes: Any) -> bytes:
    body = {**GOOD, **changes}
    return json.dumps({k: v for k, v in body.items() if v is not _DROP}).encode()


_DROP = object()


def _reason(raw: bytes) -> str:
    with pytest.raises(ProtocolError) as err:
        parse_request(raw)
    return err.value.reason


def test_a_good_submit_parses() -> None:
    sub = parse_request(_req())
    assert isinstance(sub, Submit)
    assert (sub.run_id, sub.job_kind, sub.gpu, str(sub.run_dir)) == (RUN, "train", True, f"/app/outputs/runs/{RUN}")
    node = "try-0123456789abcdef"
    graph = parse_request(_req(run_id=node, run_dir=f"/app/outputs/graph/runs/{node}", job_kind="serve",
                               inputs=["/app/uploads/" + "a" * 32 + ".png"],
                               argv=["python", "-m", "vlm_pipeline.serve.trial", "/workspace/x.json"]))  # fmt: skip
    assert isinstance(graph, Submit)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"run_id": "../../etc"}, "bad_run_id"),
        ({"run_id": "20261013-1200-UPPER"}, "bad_run_id"),
        ({"job_kind": "shell"}, "bad_job_kind"),
        ({"image_tag": "nvcr.io/x@sha256:" + "a" * 64}, "bad_image_tag"),  # a tag only: the server pins the id
        ({"image_tag": "x:1 --privileged"}, "bad_image_tag"),
        ({"argv": ["bash", "-c", "id"]}, "argv_not_allowed"),
        ({"argv": ["python", "-c", "import os"]}, "argv_not_allowed"),
        ({"argv": ["python", "-m", "vlm_pipeline.quantize.ptq"]}, "argv_not_allowed"),  # wrong kind for train
        (
            {"job_kind": "download", "max_bytes": 1, "argv": ["python", "-m", "vlm_pipeline.data.download"]},
            "argv_not_allowed",
        ),
        ({"job_kind": "download", "argv": ["python", "-m", "vlm_pipeline.download.hub"]}, "bad_max_bytes"),
        ({"max_bytes": 10}, "bad_max_bytes"),  # only a download has a byte budget
        ({"argv": ["automodel", "a\nb"]}, "bad_argv"),
        ({"argv": []}, "bad_argv"),
        ({"gpu": "yes"}, "bad_gpu"),
        ({"memory_gb": 0}, "bad_memory_gb"),
        ({"memory_gb": 512}, "bad_memory_gb"),
        ({"memory_gb": True}, "bad_memory_gb"),
        ({"lock_token": _DROP}, "gpu_needs_lock_token"),
        ({"lock_token": "short"}, "bad_lock_token"),
        ({"run_dir": "/app/outputs/runs/other-run"}, "bad_run_dir"),  # not this job's dir
        ({"run_dir": "/app/outputs"}, "bad_run_dir"),
        ({"run_dir": "/app/src"}, "bad_run_dir"),
        ({"run_dir": f"/app/outputs/runs/../runs/{RUN}"}, "bad_run_dir"),
        ({"run_dir": "/app/outputs/graph/runs/.hidden"}, "bad_run_dir"),
        ({"inputs": ["/etc/passwd"]}, "input_outside_roots"),
        ({"inputs": ["/app/.env.local"]}, "input_outside_roots"),
        ({"inputs": ["/app/src/vlm_pipeline"]}, "input_outside_roots"),
        ({"inputs": ["/app/data"]}, "input_outside_roots"),  # a whole root is too wide
        ({"inputs": ["/app/outputs/runs"]}, "input_outside_roots"),
        ({"inputs": ["/app/data/../.env"]}, "bad_inputs"),
        ({"inputs": ["/app/data/./x"]}, "bad_inputs"),
        ({"inputs": [f"/app/outputs/runs/{RUN}/checkpoints"]}, "input_overlaps_run_dir"),
        ({"inputs": "/app/data/x"}, "bad_inputs"),
        ({"run_dir": "/app/outputs/graph/runs/n4-0123456789abcdef"}, "bad_run_dir"),  # another job's node dir
        ({"inputs": ["/app/uploads/" + "a" * 32 + ".png", "/app/uploads/x.sh"]}, "bad_upload"),
        ({"inputs": ["/app/uploads/" + "a" * 32 + ".png/x"]}, "bad_upload"),
        ({"inputs": ["/app/data/ds1", "/app/data/ds1/images"]}, "inputs_nested"),
        ({"inputs": ["/app/data/ds1", "/app/data/ds1"]}, "inputs_nested"),
        ({"mounts": ["/:/host"]}, "unknown_field"),
        ({"privileged": True}, "unknown_field"),
        ({"env": {"HF_TOKEN": "x"}}, "unknown_field"),
        ({"run_dir": _DROP}, "missing_field"),
    ],
)
def test_submit_refusals_name_the_rule(changes: dict[str, Any], reason: str) -> None:
    assert _reason(_req(**changes)) == reason


def test_request_shape_refusals() -> None:
    assert _reason(b"x" * (MAX_REQUEST_BYTES + 1)) == "request_too_large"
    assert _reason(b"not json") == "not_json"
    assert _reason(b"[]") == "not_json"
    assert _reason(b'{"command": "exec", "run_id": "x"}') == "unknown_command"
    assert _reason(b'{"command": "pull"}') == "unknown_command"


def test_run_ref_commands_take_only_a_run_id() -> None:
    assert parse_request(json.dumps({"command": "stop", "run_id": RUN}).encode()) == RunRef("stop", RUN)
    logs = parse_request(json.dumps({"command": "logs", "run_id": RUN, "tail": 50}).encode())
    assert logs == RunRef("logs", RUN, 50, 0.0)
    assert _reason(json.dumps({"command": "kill", "run_id": RUN, "name": "kafka"}).encode()) == "unknown_field"
    assert _reason(json.dumps({"command": "remove", "run_id": "kafka"}).encode()) == "bad_run_id"
    assert _reason(json.dumps({"command": "logs", "run_id": RUN, "tail": 10_001}).encode()) == "bad_tail"


def test_a_refusal_never_carries_the_input() -> None:
    secret = "hf_" + "Z" * 34
    with pytest.raises(ProtocolError) as err:
        parse_request(_req(argv=["bash", secret]))
    assert secret not in str(err.value)
    assert secret not in repr(err.value)
