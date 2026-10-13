"""eval.benchmarks suite "ocrbench" on the vLLM batch (R3 inside the node): data checks, pairing, gates, base cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.catalog import load_catalog, load_dataset_catalog
from vlm_pipeline.common.local_cache import eval_data_dir
from vlm_pipeline.eval import node_worker as nw
from vlm_pipeline.eval import ocrbench
from vlm_pipeline.eval.config import EvalParams

ENGINE = {"vllm": "0.31.0", "dtype": "bfloat16", "max_model_len": 8192, "enforce_eager": True, "seed": 0,
          "sampling": {"temperature": 0.0}, "max_num_seqs": 256, "max_num_batched_tokens": 8192,
          "enable_chunked_prefill": True, "chunk": 64, "kv_cache_gb": 8.0,
          "chat_template_kwargs": {"enable_thinking": False}, "mm_processor_kwargs": {}, "image": "vlm-vllm:0.31.0",
          "container_digest": "sha256:v"}  # fmt: skip
QUESTIONS: list[dict[str, Any]] = [
    {"row": i, "image": f"images/{i:04d}.jpg", "question": f"q{i}", "answers": [f"ans{i}"], "dataset": "IIIT5K",
     "question_type": "Regular Text Recognition"}
    for i in range(10)
]  # fmt: skip


@dataclass
class FakeBatch:
    """Stands in for serve.vllm_runner.run_vllm_batch: answers per model path, records each call."""

    answers: dict[str, list[bool]]  # model dir name -> per question: correct?
    engine: dict[str, Any] = field(default_factory=lambda: dict(ENGINE))
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, root: Path, model_dir: Path, requests: list[dict[str, Any]], **kw: Any) -> Any:
        self.calls.append({"model": model_dir.name, **kw})
        run_dir: Path = kw["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        good = self.answers[model_dir.name]
        rows = [
            {"id": r["id"], "text": f"it says ans{r['id']}" if good[int(r["id"])] else "nothing",
             "finish_reason": "stop", "gen_tokens": 4}
            for r in requests
        ]  # fmt: skip
        engine = {**self.engine, "quantization": kw.get("quantization")}

        class Result:
            ok = True
            status = "completed"

            def __init__(self) -> None:
                self.run_dir = run_dir

            def rows(self) -> list[dict[str, Any]]:
                return rows

            def engine(self) -> dict[str, Any]:
                return engine

        return Result()


def _ocrbench_entry() -> Any:
    entry = load_dataset_catalog().get("ocrbench")
    assert entry is not None
    return entry


@pytest.fixture(autouse=True)
def _pin_test_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The node checks the copy against the catalog pin: point the pin at the test copy's manifest (when written)."""
    real = load_dataset_catalog()

    class Catalog:
        datasets = real.datasets

        def get(self, catalog_id: str) -> Any:
            entry = real.get(catalog_id)
            if entry is None or catalog_id != "ocrbench":
                return entry
            return entry.model_copy(update={"extracted_manifest_sha256": PIN.get("sha")})

    monkeypatch.setattr(nw, "load_dataset_catalog", Catalog)
    PIN.clear()


PIN: dict[str, str] = {}


def _data(root: Path) -> Path:
    data = eval_data_dir(root, _ocrbench_entry())
    (data / "images").mkdir(parents=True)
    lines = []
    for q in QUESTIONS:
        img = data / q["image"]
        img.write_bytes(b"\xff\xd8\xff" + bytes([q["row"]]))
        lines.append(f"{hashlib.sha256(img.read_bytes()).hexdigest()}  {q['image']}")
    (data / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    lines.append(f"{hashlib.sha256((data / 'questions.jsonl').read_bytes()).hexdigest()}  questions.jsonl")
    (data / ocrbench.MANIFEST).write_text("\n".join(lines) + "\n")
    PIN["sha"] = hashlib.sha256((data / ocrbench.MANIFEST).read_bytes()).hexdigest()
    return data


def _request(root: Path, runner: FakeBatch, *, model_hash: str, model: str = "fp8", **params: Any) -> nw.EvalRequest:
    entry = load_catalog().get("qwen3.5-2b")
    assert entry is not None
    for name in (model, "base"):
        (root / "models" / name).mkdir(parents=True, exist_ok=True)
    return nw.EvalRequest(
        root=root, node_id="ev", params=EvalParams(level="standard", **params),
        model=nw.ModelRef(root / "models" / model, model_hash),
        base=nw.ModelRef(root / "models" / "base", f"{entry.repo}@{entry.revision}"),
        entry=entry, run_dir=root / "outputs" / "graph" / "runs" / "ev-1", image="", who="t", vllm_batch=runner,
    )  # fmt: skip


def _base_hash() -> str:
    entry = load_catalog().get("qwen3.5-2b")
    assert entry is not None
    return f"{entry.repo}@{entry.revision}"


def test_fp8_is_paired_with_its_bf16_parent_and_gated_by_p002(tmp_path: Path) -> None:
    _data(tmp_path)
    runner = FakeBatch({"fp8": [True] * 9 + [False], "base": [True] * 10})  # FP8 loses one question of ten
    out = nw.run_ocrbench(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"))
    report = json.loads(out.report_path.read_text())
    assert report["quantization"] == {"precision": "fp8", "parent": _base_hash()}
    assert report["comparison"]["compare_to"] == "parent"
    assert report["comparison"]["delta"] == -10.0  # points on 0-100, as gates.yaml expects
    assert report["gate"]["status"] == "FAIL"  # fp8 below -2
    assert [c["quantization"] for c in runner.calls] == ["modelopt", None]  # model, then the bf16 parent
    assert {c["kv_cache_gb"] for c in runner.calls} == {ocrbench.KV_CACHE_GB}  # pinned for both
    variants = {r["variant"] for r in report["records"]}
    assert variants == {"model", "parent"}
    assert all(r["citation"] == ocrbench.CITATION for r in report["records"])


def test_the_parent_run_is_cached_and_reused_only_on_the_same_engine(tmp_path: Path) -> None:
    _data(tmp_path)
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    req = _request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8")
    assert nw.run_ocrbench(req).status == "WARN"  # equal scores, but n=10 < 50: quick_measurement
    nw.run_ocrbench(req)
    assert [c["model"] for c in runner.calls] == ["fp8", "base", "fp8"]  # parent from the cache the second time
    runner.engine = {**ENGINE, "container_digest": "sha256:other"}  # another vLLM image: the cache is not used
    nw.run_ocrbench(req)
    assert [c["model"] for c in runner.calls][-2:] == ["fp8", "base"]


def test_missing_or_altered_data_asks_for_the_download(tmp_path: Path) -> None:
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    req = _request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8")
    PIN["sha"] = "a" * 64  # pinned in the catalog, just not on this machine yet
    with pytest.raises(nw.EvalDataNotDownloadedError) as missing:
        nw.run_ocrbench(req)
    assert missing.value.code == "NET-002"
    assert missing.value.params["reason"] == "not_cached"
    assert (missing.value.params["catalog_id"], missing.value.params["kind"]) == ("ocrbench", "dataset")
    data = _data(tmp_path)
    (data / "images" / "0003.jpg").write_bytes(b"changed")
    with pytest.raises(nw.EvalDataNotDownloadedError) as altered:
        nw.run_ocrbench(req)
    assert altered.value.params["reason"] == "not_prepared"
    assert runner.calls == []  # nothing ran on bad data


def test_a_quantized_fine_tuned_model_is_refused_until_its_parent_can_be_run(tmp_path: Path) -> None:
    _data(tmp_path)
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    with pytest.raises(nw.EvalSuiteUnsupportedError, match="fine-tuned"):
        nw.run_ocrbench(_request(tmp_path, runner, model_hash="n3-x:n4-abc:cfg:fp8".replace(":cfg", "")))
    assert runner.calls == []


def test_a_fine_tuned_model_is_compared_with_its_base_as_a_forgetting_check(tmp_path: Path) -> None:
    _data(tmp_path)
    runner = FakeBatch({"tuned": [True] * 10, "base": [True] * 10})
    out = nw.run_ocrbench(_request(tmp_path, runner, model_hash="n5-merge:abc", model="tuned"))
    report = json.loads(out.report_path.read_text())
    assert report["quantization"] is None
    assert report["comparison"]["compare_to"] == "base"
    assert [c["quantization"] for c in runner.calls] == [None, None]


def test_support_ocrbench_for_proven_chat_models_but_never_an_unmerged_adapter() -> None:
    qwen = load_catalog().get("qwen3.5-2b")
    nw.check_support("ev", EvalParams(), qwen, {"model": "model:QuantizedModel"})
    nw.check_support("ev", EvalParams(suite="documents"), qwen, {"model": "model:Checkpoint"})
    with pytest.raises(nw.EvalSuiteUnsupportedError, match="merge"):
        nw.check_support("ev", EvalParams(), qwen, {"model": "model:Adapter"})
    with pytest.raises(nw.EvalSuiteUnsupportedError, match="thinking"):
        nw.check_support("ev", EvalParams(thinking=True), qwen, {"model": "model:Model"})


def test_a_link_planted_in_the_job_dir_is_never_written_through(tmp_path: Path) -> None:
    """B1: the parent's scores go to a host-owned cache file, never into the job's writable run dir."""
    _data(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")

    class Planting(FakeBatch):
        def __call__(self, root: Path, model_dir: Path, requests: list[dict[str, Any]], **kw: Any) -> Any:
            kw["run_dir"].mkdir(parents=True, exist_ok=True)
            (kw["run_dir"] / "scored.json").symlink_to(victim)
            return super().__call__(root, model_dir, requests, **kw)

    runner = Planting({"fp8": [True] * 10, "base": [True] * 10})
    nw.run_ocrbench(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"))
    assert victim.read_text() == "untouched"


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    _data(tmp_path)
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    req = _request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8")
    nw.run_ocrbench(req)
    (cache,) = (tmp_path / "outputs" / "eval-cache").glob("ocrbench-*.json")
    cache.write_text('{"scored": [{"row": 0, "score": 7}], "engine": {}}')
    nw.run_ocrbench(req)
    assert [c["model"] for c in runner.calls] == ["fp8", "base", "fp8", "base"]  # recomputed, no KeyError


def test_a_copy_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    """Self-consistent but different data (another extraction, an edited question + its new hash) is not OCRBench."""
    _data(tmp_path)
    PIN["sha"] = "0" * 64
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    with pytest.raises(nw.EvalDataNotDownloadedError) as err:
        nw.run_ocrbench(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"))
    assert err.value.params["reason"] == "not_prepared"


def test_without_a_catalog_pin_nothing_is_scored(tmp_path: Path) -> None:
    _data(tmp_path)
    PIN.clear()
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    with pytest.raises(nw.EvalDataNotDownloadedError) as err:
        nw.run_ocrbench(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"))
    assert err.value.params["reason"] == "not_pinned"


@pytest.mark.parametrize("line", ["not a manifest line", f"{'0' * 64}  ../../etc/passwd", f"{'0' * 64}  images/x.gif"])
def test_a_malformed_manifest_line_asks_for_a_repair(tmp_path: Path, line: str) -> None:
    data = _data(tmp_path)
    with (data / ocrbench.MANIFEST).open("a") as fh:
        fh.write(line + "\n")
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    with pytest.raises(nw.EvalDataNotDownloadedError) as err:
        nw.run_ocrbench(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"))
    assert err.value.params["reason"] == "not_prepared"
    assert runner.calls == []


def test_every_vllm_job_has_a_claimable_run_id_and_a_failed_parent_run_stays_retryable(tmp_path: Path) -> None:
    # run_job claims manifest.run_id once and refuses names outside RUN_ID: without one, the quantize_bench smoke
    # failed bad_run_id (run id taken from the dir name "model").
    import dataclasses

    from vlm_pipeline.common.run_claims import RUN_ID

    _data(tmp_path)
    runner = FakeBatch({"fp8": [True] * 10, "base": [True] * 10})
    node_dir = tmp_path / "outputs" / "graph" / "runs" / "n4-3aaefa737788c698-a2"
    req = dataclasses.replace(_request(tmp_path, runner, model_hash=f"n3-x:{_base_hash()}:fp8"), run_dir=node_dir)
    nw.run_ocrbench(req)
    for cache in (tmp_path / "outputs" / "eval-cache").glob("ocrbench-*.json"):
        cache.unlink()  # as if the parent run had failed: it must run again under a NEW claim
    nw.run_ocrbench(req)
    ids = [c["manifest"].run_id for c in runner.calls]
    assert ids[0] == ids[2] == "n4-3aaefa737788c698-a2-model"
    assert ids[1] != ids[3]
    assert all(RUN_ID.fullmatch(i) for i in ids), ids
    assert all(c["manifest"].stage == "eval" for c in runner.calls)
