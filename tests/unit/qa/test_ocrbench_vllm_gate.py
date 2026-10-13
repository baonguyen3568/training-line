"""R3 gate driver: two runs pair only when questions, data copy and engine settings match; the verdict is P002's."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "ocrbench_vllm_gate.py"


def _mod() -> Any:
    spec = importlib.util.spec_from_file_location("ocrbench_vllm_gate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = {"vllm": "0.31.0", "dtype": "bfloat16", "max_model_len": 8192, "enforce_eager": True, "seed": 0,
          "sampling": {"temperature": 0.0, "n": 1},
          "max_num_seqs": 256, "max_num_batched_tokens": 8192, "enable_chunked_prefill": True, "chunk": 64,
          "chat_template_kwargs": {"enable_thinking": False},
          "mm_processor_kwargs": {"min_pixels": 200704, "max_pixels": 1003520},
          "container_digest": "sha256:a"}  # fmt: skip


def _run(label: str, scores: list[int], **engine: Any) -> dict[str, Any]:
    return {
        "label": label,
        "rows": [{"row": i, "score": s} for i, s in enumerate(scores)],
        "data_manifest_sha256": "m",
        "max_tokens": 256,
        "engine": {**ENGINE, **engine},
        "truncated_pct": 0.0,
        "accuracy": 100 * sum(scores) / len(scores),
    }


def test_runs_pair_only_on_equal_questions_data_and_engine() -> None:
    mod = _mod()
    a = _run("bf16", [1, 0, 1])
    assert mod.comparable(a, _run("fp8", [1, 1, 0], quantization="modelopt")) == []  # quantization may differ
    assert mod.comparable(a, _run("x", [1, 0, 1], chunk=32)) == ["engine chunk: 64 != 32"]
    other = _run("x", [1, 0, 1])
    other["data_manifest_sha256"] = "n"
    assert "different data copy (manifest)" in mod.comparable(a, other)
    shorter = _run("x", [1, 0])
    assert "different questions or order" in mod.comparable(a, shorter)


def test_compare_reports_the_paired_delta_noise_and_p002_verdict(tmp_path: Path, capsys: Any) -> None:
    mod = _mod()
    n = 500
    ref = _run("bf16-a", [1] * 400 + [0] * 100)
    noise = _run("bf16-b", [1] * 399 + [0] * 101)
    fp8 = _run("fp8", [1] * 380 + [0] * 120)  # -4 points: FAIL for fp8 (< -2)
    paths = {}
    for name, run in (("ref", ref), ("noise", noise), ("fp8", fp8)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(run))
    mod.main(["compare", str(paths["ref"]), str(paths["fp8"]), "--precision", "fp8", "--noise", str(paths["noise"])])
    report = json.loads(capsys.readouterr().out)
    assert report["paired"]["delta"] == pytest.approx(-4.0)
    assert report["paired"]["n"] == n
    assert report["noise"]["changed_answers"] == 1
    assert report["gate"]["status"] == "FAIL"


def test_incomparable_runs_are_refused(tmp_path: Path) -> None:
    mod = _mod()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run("bf16", [1, 0])))
    b.write_text(json.dumps(_run("fp8", [1, 0], max_num_seqs=8)))
    with pytest.raises(SystemExit, match="not comparable: engine max_num_seqs"):
        mod.main(["compare", str(a), str(b), "--precision", "fp8"])


def test_scores_pair_by_question_id_not_position() -> None:
    mod = _mod()
    ref = _run("bf16", [1, 0, 1])
    cand = _run("fp8", [0, 1, 1])
    cand["rows"] = list(reversed(cand["rows"]))  # same ids, other order: pairing must follow ids
    assert mod.paired_scores(ref, cand) == ([0, 1, 1], [1, 0, 1])
    cand["rows"][0]["row"] = 99
    with pytest.raises(SystemExit, match="different questions"):
        mod.paired_scores(ref, cand)


def test_sampling_must_match() -> None:
    mod = _mod()
    hot = _run("x", [1], sampling={"temperature": 0.7, "n": 1})
    assert any(p.startswith("engine sampling") for p in mod.comparable(_run("bf16", [1]), hot))
