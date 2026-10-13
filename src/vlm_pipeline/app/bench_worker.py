"""Host side of the speed benchmark node (``bench.perf``, M1b B-6; protocol: skill tester-perf-bench).

One container job per measured model through ``run_job`` (preflight, GPU lock, memguard, on_wait); the container side
is ``vlm_pipeline.serve.bench``, which times the "Chạy thử" code path. What to load is resolved exactly as for
"Chạy thử" (``trial_worker``: catalog base, base processor + generation config, template decode_overrides), so the
benchmark measures the decoding the user gets. ``compare_with=base`` also measures the catalog base (second job,
same images and conditions) and reports the ratio. Never cached (``cacheable=False``): timings depend on host load.
Only Nemotron-Parse on engine hf for now; chat models need the vLLM server (B7-3), multi-user needs batching there.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from vlm_pipeline.app.trial_worker import _template_overrides, base_of, est_peak_gb
from vlm_pipeline.common.catalog import CatalogEntry, load_dataset_catalog
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.jobrun import ContainerJob, MemoryPreemptedBeforeStartError, run_job
from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.serve.bench import BenchRequest
from vlm_pipeline.serve.config import BenchParams
from vlm_pipeline.serve.trial import TrialRequest

if TYPE_CHECKING:
    from vlm_pipeline.app.workers import RealWorker

REPORT_SCHEMA_VERSION = 1
BENCH_IMAGES = 10  # first CORD-v2 test images, cycled over num_requests (fixed set: runs compare)
WARMUP = 3
COMPARED_METRICS = ("e2e_ms_p50", "tokens_per_s", "memory_peak_load_gb")


class BenchUnsupportedError(VlmPipelineError):
    """This model / scenario has no benchmark engine yet (shown before the GPU lock)."""

    code: ClassVar[str] = "CFG-011"


def _check_supported(entry: CatalogEntry, params: BenchParams, node_id: str) -> None:
    reason = None
    if entry.loss != "nemotron_parse":
        reason = "chat models are benchmarked on the vLLM server (not available yet)"
    elif params.scenario != "single_user":
        reason = "multi-user needs a serving engine with batching (vLLM, not available yet)"
    elif params.compare_with == "unquantized":
        reason = "the unquantized parent comparison comes with the quantize node"
    if reason:
        raise BenchUnsupportedError(reason, params={"node": node_id, "node_type": "bench.perf"})


def _cord_dir(worker: RealWorker) -> Path:
    entry = load_dataset_catalog().get("cord-v2")
    if entry is None:
        raise BenchUnsupportedError("cord-v2 is not in the dataset catalog", params={"node_type": "bench.perf"})
    repo_dir = f"datasets--{entry.repo.replace('/', '--')}"
    return worker.root / ".cache" / "hf" / "hub" / repo_dir / "snapshots" / entry.revision


def _trial(worker: RealWorker, model_dir: Path, base: Path, entry: CatalogEntry, params: BenchParams) -> TrialRequest:
    return TrialRequest(
        model_dir=worker.container_path(model_dir),
        generation_config_dir=worker.container_path(base),
        processor_dir=worker.container_path(base),
        family="nemotron_parse",
        image="",  # images come from the benchmark set
        prompt="",
        max_new_tokens=params.output_tokens,
        temperature=0.0,
        top_p=1.0,
        attn=entry.attn,
        decode_overrides=_template_overrides(worker, entry),
    )


def _measure(worker: RealWorker, job: NodeJob, entry: CatalogEntry, trial: TrialRequest, name: str) -> dict[str, Any]:
    params = job.params
    assert isinstance(params, BenchParams)  # noqa: S101 - registry guarantees the params model
    run_dir = job.run_dir.resolve() / "jobs" / name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").unlink(missing_ok=True)
    request = BenchRequest(
        trial=asdict(trial),
        data_dir=worker.container_path(_cord_dir(worker)),
        image_ids=list(range(BENCH_IMAGES)),
        num_requests=params.num_requests,
        warmup=WARMUP,
    )
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(asdict(request), indent=1), encoding="utf-8")
    run_id = f"{job.run_dir.name[:100]}-{name}"
    manifest = RunManifest(
        run_id=run_id,
        name=f"bench.perf {entry.display_name} ({name})",
        stage="bench",
        task="bench",
        base_model=entry.repo,
        base_model_revision=entry.revision,
        container_image=worker.image,
    )
    container = ContainerJob(
        name=run_id,
        image=worker.image,
        argv=("python", "-m", "vlm_pipeline.serve.bench", worker.container_path(request_path)),
        read_only_project=True,  # same isolation as "Chạy thử" (remote model code): only this job's dir is writable
        rw_dirs=(run_dir,),
        env={"HF_MODULES_CACHE": "/tmp/hf_modules", "TRITON_CACHE_DIR": "/tmp/triton"},  # noqa: S108
    )
    try:
        final = run_job(
            container,
            manifest,
            run_dir,
            project_root=worker.root,
            who=worker.who,
            est_peak_gb=est_peak_gb(entry),
            cancel=job.cancel,
            require_approved_image=True,
            on_wait=job.on_wait,
        )
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc), params={"need_gb": est_peak_gb(entry)}) from exc
    result_path = run_dir / "result.json"
    if final.status != RunStatus.COMPLETED or not result_path.is_file():
        from vlm_pipeline.app.workers import _tail

        raise ContainerExit(
            75 if final.status == RunStatus.INTERRUPTED else (final.exit_code or 1),
            oom_killed=bool(final.oom_killed),
            cancelled_by_executor=final.status == RunStatus.CANCELLED,
            log_tail=_tail(run_dir / "logs" / "stdout.log"),
        )
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_id"] = run_id
    result["container_digest"] = final.container_digest
    return result


def records(result: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    """Flat perf records (frontend-metrics-contract: benchmark="perf", conditions = perf_conditions)."""
    skip = {
        "per_request",
        "decoding",
        "overrides",
        "model_class",
        "conditions",
        "request",
        "bench_sha256",
        "run_id",
        "container_digest",
    }
    conditions = {
        **result["conditions"],
        "output_tokens": result["decoding"].get("max_new_tokens"),
        "host_baseline_available_gb": result.get("host_baseline_available_gb"),
        # what makes two measurements comparable: the same images and the same measuring code
        "image_ids": result.get("request", {}).get("image_ids"),
        "bench_sha256": result.get("bench_sha256"),
    }
    return [
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "benchmark": "perf",
            "variant": variant,
            "metric": k,
            "value": v,
            "perf_conditions": conditions,
            "container_digest": result.get("container_digest"),
            "source": "measured",
        }
        for k, v in result.items()
        if k not in skip and isinstance(v, int | float)
    ]


def run_bench(worker: RealWorker, job: NodeJob) -> NodeResult:
    params = job.params
    assert isinstance(params, BenchParams)  # noqa: S101
    ref = job.inputs["model"]
    entry, base = base_of(worker, ref)
    _check_supported(entry, params, job.node_id)
    model_dir = worker.host_path(ref.path)
    results = {"model": _measure(worker, job, entry, _trial(worker, model_dir, base, entry, params), "model")}
    is_base = ref.hash == f"{entry.repo}@{entry.revision}"
    if params.compare_with == "base" and not is_base:
        results["base"] = _measure(worker, job, entry, _trial(worker, base, base, entry, params), "base")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "bench_report",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": {"path": str(model_dir), "hash": ref.hash},
        "records": [r for variant, res in results.items() for r in records(res, variant)],
        "runs": {
            v: {
                k: res[k]
                for k in (
                    "run_id",
                    "container_digest",
                    "bench_sha256",
                    "decoding",
                    "overrides",
                    "conditions",
                    "model_class",
                )
            }
            for v, res in results.items()
        },
    }
    if "base" in results:
        report["ratio_vs_base"] = {
            m: round(results["model"][m] / results["base"][m], 3) for m in COMPARED_METRICS if results["base"].get(m)
        }
    path = job.run_dir.resolve() / "report.json"
    text = json.dumps(report, indent=1) + "\n"
    path.write_text(text, encoding="utf-8")
    port = next(iter(job.spec.outputs))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return NodeResult(outputs={port: ArtifactRef(type="BenchReport", path=str(path), hash=f"sha256:{digest}")})


__all__ = ["BenchUnsupportedError", "records", "run_bench"]
