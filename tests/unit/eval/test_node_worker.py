"""eval.benchmarks engine (B-5) without GPU: run_job is replaced by a fake that writes predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.catalog import load_catalog
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.eval import node_worker as nw
from vlm_pipeline.eval.config import EvalParams

GT = {"menu": {"nm": "Tea", "price": "5"}}
GOOD = "<s_menu><s_nm>Tea</s_nm><s_price>5</s_price></s_menu>"
HALF = "<s_menu><s_nm>Tea</s_nm><s_price>6</s_price></s_menu>"


class FakeJobs:
    """Stands in for common.jobrun.run_job: records each job, writes pred.jsonl like eval_cord_worker."""

    def __init__(self, sequences: dict[str, str], status: RunStatus = RunStatus.COMPLETED) -> None:
        self.sequences = sequences  # model dir name -> sequence for every image
        self.status = status
        self.jobs: list[tuple[str, ...]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        self.kwargs.append(kw)
        argv = job.argv
        self.jobs.append(argv)
        model = Path(argv[argv.index("--model") + 1]).name
        ids = [int(i) for i in argv[argv.index("--ids") + 1].split(",")]
        rows = [
            {"index": i, "sequence": self.sequences[model], "gt_parse": GT, "truncated": False, "split": "test"}
            for i in ids
        ]
        (run_dir / "pred.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        final = manifest.model_copy(
            update={"status": self.status, "exit_code": 0 if self.status == RunStatus.COMPLETED else 1}
        )
        (run_dir / "run.json").write_text(final.model_dump_json())
        return final


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "scripts" / "qa").mkdir(parents=True)
    (tmp_path / "scripts" / "qa" / "eval_cord_worker.py").write_text("# worker v1\n")
    for name in ("base", "finetuned"):
        (tmp_path / "models" / name).mkdir(parents=True)
    monkeypatch.setattr(nw, "image_digest", lambda image: "sha256:img")
    _template(tmp_path, "decode_overrides:\n  repetition_penalty: 1.0\n")
    return tmp_path


def _template(root: Path, text: str) -> None:
    path = root / "configs" / "train" / "templates" / "nemotron-parse-v1.1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _flags(argv: tuple[str, ...]) -> dict[str, str]:
    return {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--") and not argv[i + 1].startswith("--")}


def _request(root: Path, *, model: str = "finetuned", **params: Any) -> nw.EvalRequest:
    entry = load_catalog().get("nemotron-parse-v1.1")
    assert entry is not None
    return nw.EvalRequest(
        root=root,
        node_id="eval",
        params=EvalParams(**params),
        model=nw.ModelRef(root / "models" / model, f"{model}-hash"),
        base=nw.ModelRef(root / "models" / "base", "base-hash"),
        entry=entry,
        run_dir=root / "outputs" / "runs" / "node-eval-abc",
        image="vlm-automodel:test",
        who="vlm-app",
    )


def test_finetune_beats_base_report_and_gate(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    outcome = nw.run_eval_benchmarks(_request(root, level="standard"))
    report = json.loads(outcome.report_path.read_text())
    assert outcome.report_path == root / "outputs" / "runs" / "node-eval-abc" / "report.json"
    assert report["comparison"]["delta"] == 50.0  # 100 vs 50: HALF = 1 hit, 1 wrong + 1 missing field
    assert report["comparison"]["ci95"][0] > 0
    assert (outcome.status, report["gate"]["status"]) == ("PASS", "PASS")
    f1 = {(r["variant"], r["metric"]): r["value"] for r in report["records"]}
    assert (f1["model", "field_f1"], f1["base", "field_f1"]) == (100.0, 50.0)
    assert len(report["image_ids"]) == 100
    assert {r["n_samples"] for r in report["records"]} == {100}


def test_generation_follows_the_p3_protocol(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": GOOD})
    monkeypatch.setattr(nw, "run_job", jobs)
    nw.run_eval_benchmarks(_request(root, compare_with="none", max_new_tokens=512))
    [argv] = jobs.jobs
    flags = _flags(argv)
    assert flags["--repetition-penalty"] == "1.0"  # the template's decode_overrides, same as "Chạy thử"
    assert flags["--use-cache"] == "true"
    assert flags["--batch-size"] == "1"
    assert flags["--max-new-tokens"] == "512"
    assert flags["--expect-prefix"] == "2,0,50004,50008,50001"
    assert flags["--generation-config"] == "/workspace/models/base"  # always the base's decoding defaults
    assert "--require-no-drift" in argv
    assert len(flags["--ids"].split(",")) == 30  # level quick


def test_quick_level_warns_and_base_is_cached(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    first = nw.run_eval_benchmarks(_request(root))
    assert first.status == "WARN"  # 30 images: indicative only
    reasons = json.loads(first.report_path.read_text())["gate"]["reasons"]
    assert [r["code"] for r in reasons] == ["quick_measurement"]
    nw.run_eval_benchmarks(_request(root))
    models = [Path(a[a.index("--model") + 1]).name for a in jobs.jobs]
    assert models == ["finetuned", "base", "finetuned"]  # second run reused the cached base predictions
    assert list((root / "outputs" / "eval-cache").iterdir())  # cache lives outside the node run dir


def test_base_cache_misses_when_the_worker_changes(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    nw.run_eval_benchmarks(_request(root))
    (root / "scripts" / "qa" / "eval_cord_worker.py").write_text("# worker v2\n")
    nw.run_eval_benchmarks(_request(root))
    assert [Path(a[a.index("--model") + 1]).name for a in jobs.jobs].count("base") == 2


def test_evaluating_the_base_itself_has_no_comparison(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nw, "run_job", FakeJobs({"base": GOOD}))
    request = _request(root, model="base", level="standard")
    request = nw.EvalRequest(**{**request.__dict__, "model": request.base})
    report = json.loads(nw.run_eval_benchmarks(request).report_path.read_text())
    assert report["comparison"] is None
    assert {r["variant"] for r in report["records"]} == {"model"}


def test_failed_job_raises_with_its_exit(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nw, "run_job", FakeJobs({"finetuned": GOOD}, status=RunStatus.FAILED))
    with pytest.raises(nw.EvalJobFailed) as exc:
        nw.run_eval_benchmarks(_request(root))
    assert (exc.value.exit_code, exc.value.cancelled) == (1, False)


@pytest.mark.parametrize("params", [{"suite": "documents"}, {"thinking": True}])
def test_unsupported_requests_fail_before_any_job(
    root: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    jobs = FakeJobs({})
    monkeypatch.setattr(nw, "run_job", jobs)
    with pytest.raises(nw.EvalSuiteUnsupportedError) as exc:
        nw.run_eval_benchmarks(_request(root, **params))
    assert exc.value.code == "CFG-011"
    assert jobs.jobs == []


def test_paired_f1_refuses_different_images() -> None:
    a = [{"index": 0, "sequence": GOOD, "gt_parse": GT}]
    b = [{"index": 1, "sequence": GOOD, "gt_parse": GT}]
    with pytest.raises(ValueError, match="same images"):
        nw.paired_f1(a, b)


def test_without_template_overrides_the_base_decoding_applies(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _template(root, "recipe: x\n")
    jobs = FakeJobs({"finetuned": GOOD})
    monkeypatch.setattr(nw, "run_job", jobs)
    report = json.loads(nw.run_eval_benchmarks(_request(root, compare_with="none")).report_path.read_text())
    assert "--repetition-penalty" not in jobs.jobs[0]  # the worker keeps the base generation_config value
    assert report["decoding"]["template_overrides"] == {}


def test_template_override_the_worker_cannot_apply_is_refused(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _template(root, "decode_overrides:\n  no_repeat_ngram_size: 3\n")
    jobs = FakeJobs({})
    monkeypatch.setattr(nw, "run_job", jobs)
    with pytest.raises(nw.EvalSuiteUnsupportedError, match="no_repeat_ngram_size"):
        nw.run_eval_benchmarks(_request(root))
    assert jobs.jobs == []


def test_changing_the_template_decoding_misses_the_base_cache(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    nw.run_eval_benchmarks(_request(root))
    _template(root, "decode_overrides:\n  repetition_penalty: 1.1\n")
    nw.run_eval_benchmarks(_request(root))
    assert [Path(a[a.index("--model") + 1]).name for a in jobs.jobs].count("base") == 2


def test_memory_wait_callback_reaches_both_jobs(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    events: list[dict[str, float]] = []
    request = nw.EvalRequest(**{**_request(root).__dict__, "on_wait": events.append})
    nw.run_eval_benchmarks(request)
    assert [kw.get("on_wait") for kw in jobs.kwargs] == [events.append, events.append]  # model + base


def test_base_cache_is_written_privately_then_renamed(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    seen: list[Path] = []
    real = jobs.__call__

    def spy(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.append(run_dir)
        return real(job, manifest, run_dir, **kw)

    monkeypatch.setattr(nw, "run_job", spy)
    nw.run_eval_benchmarks(_request(root))
    base_dir = seen[1]
    assert ".tmp-" in base_dir.name  # never generated in the shared directory itself
    [entry] = list((root / "outputs" / "eval-cache").iterdir())
    assert ".tmp-" not in entry.name
    assert (entry / "pred.jsonl").is_file()


def test_failed_cache_entry_is_set_aside_not_reused(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = FakeJobs({"finetuned": GOOD, "base": HALF})
    monkeypatch.setattr(nw, "run_job", jobs)
    nw.run_eval_benchmarks(_request(root))
    [entry] = list((root / "outputs" / "eval-cache").iterdir())
    (entry / "pred.jsonl").write_text("")  # corrupt the shared entry
    nw.run_eval_benchmarks(_request(root))
    names = sorted(p.name for p in (root / "outputs" / "eval-cache").iterdir())
    assert any(".stale-" in n for n in names)
    assert [Path(a[a.index("--model") + 1]).name for a in jobs.jobs].count("base") == 2


# ------------------------------------------------------------------- check_support (validate == worker)


def _parse() -> Any:
    entry = load_catalog().get("nemotron-parse-v1.1")
    assert entry is not None
    return entry


@pytest.mark.parametrize(
    ("params", "ok"),
    [({}, True), ({"suite": "cord_test", "level": "full"}, True), ({"suite": "documents"}, False),
     ({"thinking": True}, False)],
)  # fmt: skip
def test_check_support_is_what_the_worker_enforces(
    root: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any], ok: bool
) -> None:
    """Invariant: when check_support passes, the worker does not refuse; when it raises, the worker raises the same."""
    jobs = FakeJobs({"finetuned": GOOD, "base": GOOD})
    monkeypatch.setattr(nw, "run_job", jobs)

    def check() -> None:
        nw.check_support("eval", EvalParams(**params), _parse(), {"model": "model"}, root=root)

    if ok:
        check()
        nw.run_eval_benchmarks(_request(root, **params))
    else:
        with pytest.raises(nw.EvalSuiteUnsupportedError) as at_validate:
            check()
        with pytest.raises(nw.EvalSuiteUnsupportedError) as at_run:
            nw.run_eval_benchmarks(_request(root, **params))
        assert (at_validate.value.code, at_validate.value.params) == (at_run.value.code, at_run.value.params)


def test_check_support_refuses_chat_models_and_ignores_unwired_model(root: Path) -> None:
    qwen = load_catalog().get("qwen3.5-0.8b")
    with pytest.raises(nw.EvalSuiteUnsupportedError):
        nw.check_support("eval", EvalParams(), qwen, {"model": "model"}, root=root)
    nw.check_support("eval", EvalParams(suite="documents"), None, {}, root=root)  # CFG-003 is validate's job


def test_check_support_sees_template_overrides_without_running(root: Path) -> None:
    _template(root, "decode_overrides:\n  num_beams: 2\n")
    with pytest.raises(nw.EvalSuiteUnsupportedError, match="num_beams"):
        nw.check_support("eval", EvalParams(), _parse(), {"model": "model"}, root=root)
