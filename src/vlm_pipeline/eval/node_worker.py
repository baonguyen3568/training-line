"""Engine of node ``eval.benchmarks`` (B-5): CORD-v2 accuracy of a Nemotron-Parse model, optionally paired vs its base.

Called by the app's RealWorker (``_eval``), which adapts NodeJob/NodeResult and maps ``EvalJobFailed`` to
``ContainerExit``: this module sits in the ``eval`` layer and only imports ``common`` (ADR-004 A3).

Decoding (001a §10.7): the base model's generation config (``--require-no-drift``), then the training template's
``decode_overrides`` (Parse: repetition_penalty 1.0), then the node's max_new_tokens; greedy, batch 1, use_cache on.
"Chạy thử" applies the same template overrides, so the answer the user tries is the one that was scored.
Each generation is one container job through ``run_job`` (preflight, GPU lock, memguard, run.json) in
``<run_dir>/jobs/model``. The base run, when ``compare_with=base``, is
cached under ``outputs/eval-cache/<key>`` (key = base hash + images + decoding + worker + image id), not in the node
run dir, which retention cleans per node. Output: ``<run_dir>/report.json`` (records + paired comparison + gate).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import stat
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from vlm_pipeline.common.catalog import CatalogEntry, load_dataset_catalog
from vlm_pipeline.common.decoding import decode_overrides_of
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.host_paths import hub_repo_dir
from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    MemoryPreemptedBeforeStartError,
    image_digest,
    run_job,
)
from vlm_pipeline.common.local_cache import eval_data_dir
from vlm_pipeline.common.paths import configs_root, project_root
from vlm_pipeline.common.run_log import log_tail
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest, read_small_json
from vlm_pipeline.eval import cord, ocrbench
from vlm_pipeline.eval.config import EvalParams
from vlm_pipeline.eval.gate import BenchmarkResult, finetune_gate, quant_gate

REPORT_SCHEMA_VERSION = 1
CORD_DATASET_ID = "cord-v2"
CORD_WORKER = "scripts/qa/eval_cord_worker.py"
PARSE_PREFIX = "2,0,50004,50008,50001"  # 001a §10.5: the training decoder prefix, asserted per row
LEVEL_SAMPLES = {"quick": 30, "standard": 100, "full": 100}  # CORD-v2 test has 100 images
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
FIXED_DECODING = {"batch_size": 1, "use_cache": True}
TEMPLATE_DIR = Path("configs") / "train" / "templates"
SUPPORTED_OVERRIDES = frozenset({"repetition_penalty", "max_new_tokens"})  # what eval_cord_worker can apply
# Chat models whose bf16 and FP8 exports vLLM 0.31 loads and answers (R3 gate 2026-10-13; parity smoke 0.8B): suite
# "ocrbench" (documents) runs on the vLLM batch. Others are refused at validate until measured.
OCRBENCH_VLLM_MODELS = frozenset({"qwen3.5-0.8b", "qwen3.5-2b"})
OCRBENCH_CATALOG_ID = "ocrbench"
# Quantized artifact hash "<run_id>:<parent hash>:<format>" (app/quantize_worker): format -> gates.yaml precision key.
QUANT_PRECISION = {"fp8": "fp8", "nvfp4": "nvfp4"}


class EvalSuiteUnsupportedError(VlmPipelineError):
    """The requested suite / model has no evaluation engine yet (only CORD on Nemotron-Parse so far)."""

    code: ClassVar[str] = "CFG-011"


class EvalDataNotDownloadedError(VlmPipelineError):
    """The benchmark's eval-only data is not on this machine (or not extracted): download it first (G-5)."""

    code: ClassVar[str] = "NET-002"


class EvalJobFailed(VlmPipelineError):  # noqa: N818 - mirrors graph.errors.ContainerExit, which the app raises
    """A generation container did not complete; the app turns this into ``ContainerExit``."""

    code: ClassVar[str] = "SYS-003"

    def __init__(self, final: RunManifest, log_tail: str) -> None:
        super().__init__(f"eval job {final.run_id} ended {final.status}", params={"exit_code": final.exit_code})
        self.final = final
        self.log_tail = log_tail

    @property
    def exit_code(self) -> int:
        return 75 if self.final.status == RunStatus.INTERRUPTED else (self.final.exit_code or 1)

    @property
    def cancelled(self) -> bool:
        return self.final.status == RunStatus.CANCELLED


@dataclass(frozen=True)
class ModelRef:
    """A model on the host: HF directory + content hash (the ArtifactRef fields the engine needs)."""

    path: Path
    hash: str


@dataclass(frozen=True)
class EvalRequest:
    root: Path
    node_id: str
    params: EvalParams
    model: ModelRef
    base: ModelRef  # the catalog base the model was trained from (== model when evaluating the base itself)
    entry: CatalogEntry  # catalog entry of the base
    run_dir: Path
    image: str
    who: str
    cancel: threading.Event | None = None
    # serve.vllm_runner.run_vllm_batch, injected by the app (eval may not import serve: ADR-004 A3 layers).
    vllm_batch: Callable[..., Any] | None = None
    # "Waiting for free memory" callback of the executor (run_job queues a job that does not fit yet), shown on the UI.
    on_wait: Callable[[dict[str, float]], None] | None = None


@dataclass(frozen=True)
class EvalOutcome:
    report_path: Path
    report_hash: str
    status: str  # gate: PASS | WARN | FAIL


# ----------------------------------------------------------------------------------------------- plan
def check_support(
    node_id: str,
    params: EvalParams,
    model: CatalogEntry | None,
    inputs: Mapping[str, str],
    *,
    root: Path | None = None,
) -> None:
    """Pure "is this supported" check, shared by graph validate and the worker.

    Invariant: validate clean => the worker never refuses for support reasons. Reads only versioned repo config
    (the template's decode_overrides), never user files or run outputs.

    ``inputs``: wired ports -> "dataset:hub" | "dataset:local" | "model" | "other". Raises CFG-011 when unsupported.
    """
    # cord_test / ocrbench read their own heldout data; a wired dataset matters once suite my_val exists. The model
    # port's kind is "model:<artifact type>": vLLM (ocrbench) loads full weights only, never a LoRA adapter.
    model_kind = inputs.get("model", "")
    if model is None:  # no model wired: CFG-003 from validate, nothing to say about support
        return

    def refuse(why: str) -> EvalSuiteUnsupportedError:
        return EvalSuiteUnsupportedError(why, params={"node": node_id, "node_type": "eval.benchmarks"})

    suite = suite_of(params, model)
    if suite is None:
        raise refuse(f"suite {params.suite!r} is not available for {model.id} yet")
    if params.thinking:
        raise refuse("thinking mode is not evaluated yet (answers are scored without a reasoning block)")
    # Defensive: an Adapter port does not type-check into eval.model on the canvas, so this is a second line only.
    if suite == "ocrbench" and model_kind == "model:Adapter":
        raise refuse("a LoRA adapter is scored after merging it: wire it through train.merge_lora first")
    if suite == "cord_test":
        unsupported = set(template_overrides(root or project_root(), model)) - SUPPORTED_OVERRIDES
        if unsupported:  # never evaluate with a decoding other than the one "Chạy thử" uses
            raise refuse(f"decode_overrides {sorted(unsupported)} not supported by the CORD eval worker")


def suite_of(params: EvalParams, model: CatalogEntry) -> str | None:
    """The suite this node runs for ``model`` (None = not available).

    CORD test for Nemotron-Parse; OCRBench (documents) on the vLLM batch for the chat models vLLM is proven on.
    """
    if model.loss == "nemotron_parse":
        return "cord_test" if params.suite in ("auto", "cord_test") else None
    if model.id in OCRBENCH_VLLM_MODELS and params.suite in ("auto", "documents"):
        return "ocrbench"
    return None


def _cord_snapshot(root: Path) -> tuple[Path, str]:
    entry = load_dataset_catalog().get(CORD_DATASET_ID)
    if entry is None:
        raise EvalSuiteUnsupportedError(
            "cord-v2 is not in the dataset catalog", params={"node_type": "eval.benchmarks"}
        )
    path = root / ".cache" / "hf" / "hub" / f"datasets--{entry.repo.replace('/', '--')}" / "snapshots" / entry.revision
    return path, entry.revision


def _container(root: Path, host: Path) -> str:
    return f"{CONTAINER_ROOT}/{host.resolve().relative_to(root.resolve()).as_posix()}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_overrides(root: Path, entry: CatalogEntry) -> dict[str, Any]:
    """The training template's ``decode_overrides`` (base -> template -> node, 001a §10.7); {} without a template."""
    path = configs_root(root) / TEMPLATE_DIR / f"{entry.id}.yaml"
    if not path.is_file():
        return {}
    return decode_overrides_of(yaml.safe_load(path.read_text(encoding="utf-8")) or {}).as_dict()


def decoding_of(request: EvalRequest) -> dict[str, Any]:
    """Effective decoding knobs this node controls (recorded in the report and in the base cache key)."""
    overrides = template_overrides(request.root, request.entry)
    return {
        **FIXED_DECODING,
        "template_overrides": overrides,
        "max_new_tokens": request.params.max_new_tokens,  # node param wins over the template (base -> template -> node)
    }


def base_cache_key(base_hash: str, ids: Sequence[int], decoding: dict[str, Any], worker_sha: str, image_id: str) -> str:
    """Everything that determines the base predictions; a change in any of them is a different cache entry."""
    material = json.dumps(
        {"base": base_hash, "ids": list(ids), "decoding": decoding,
         "worker": worker_sha, "image": image_id},
        sort_keys=True,
    )  # fmt: skip
    return hashlib.sha256(material.encode()).hexdigest()[:24]


# ------------------------------------------------------------------------------------------- generate
def _generate(request: EvalRequest, model: ModelRef, ids: Sequence[int], run_dir: Path, run_id: str) -> Path:
    """One generation job (container) writing ``run_dir/pred.jsonl``; raises EvalJobFailed when it does not complete."""
    root = request.root
    data_dir, revision = _cord_snapshot(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    argv: tuple[str, ...] = (
        "python", f"{CONTAINER_ROOT}/{CORD_WORKER}",
        "--model", _container(root, model.path),
        "--data", _container(root, data_dir),
        "--split", "test",
        "--ids", ",".join(str(i) for i in ids),
        "--device", "cuda",
        "--max-new-tokens", str(request.params.max_new_tokens),
        "--batch-size", "1",
        "--expect-prefix", PARSE_PREFIX,
        "--generation-config", _container(root, request.base.path),
        "--require-no-drift",
        "--use-cache", "true",
        "--out", _container(root, run_dir / "pred.jsonl"),
    )  # fmt: skip
    penalty = decoding_of(request)["template_overrides"].get("repetition_penalty")
    if penalty is not None:  # unset: the base generation config's own value (worker default)
        argv += ("--repetition-penalty", str(penalty))
    manifest = RunManifest(
        run_id=run_id,
        name=f"eval.benchmarks {request.entry.display_name} ({run_id})",
        stage="eval",
        task="eval",
        base_model=request.entry.repo,
        base_model_revision=request.entry.revision,
        dataset=f"{CORD_DATASET_ID}:test",
        dataset_revision=revision,
        container_image=request.image,
    )
    try:
        final = run_job(
            ContainerJob(
                name=manifest.run_id,
                image=request.image,
                argv=argv,
                # P017 step 6: only its own dir writable; model, base, CORD and the worker script read-only.
                narrow=True,
                rw_dirs=(run_dir,),
                code_files=(CORD_WORKER,),
                ro_inputs=tuple(hub_repo_dir(root, p) for p in (model.path, request.base.path, data_dir)),
            ),
            manifest,
            run_dir,
            project_root=root,
            who=request.who,
            # No measured eval peak in the catalog yet (ram_gb_eval pending): the train peak is a safe upper bound.
            est_peak_gb=request.entry.ram_gb_train,
            cancel=request.cancel,
            on_wait=request.on_wait,
            require_approved_image=True,
        )
    except MemoryPreemptedBeforeStartError as exc:
        raise InsufficientMemoryError(str(exc), params={"need_gb": request.entry.ram_gb_train}) from exc
    if final.status != RunStatus.COMPLETED:
        raise EvalJobFailed(final, log_tail(run_dir))
    return run_dir / "pred.jsonl"


def _cache_hit(cache_dir: Path, n: int) -> Path | None:
    pred = cache_dir / "pred.jsonl"
    if cache_dir.is_symlink() or not pred.is_file() or not (cache_dir / "run.json").is_file():
        return None
    try:
        if read_manifest(cache_dir).status == RunStatus.COMPLETED and len(_load(pred)) == n:
            return pred
    except (OSError, ValueError):  # unreadable / linked / oversized / malformed: a miss, recomputed
        return None
    return None


def _cached_base(request: EvalRequest, ids: Sequence[int]) -> Path:
    """Base predictions from the shared cache, or generated into a private dir then renamed into place atomically.

    Two eval nodes on the same base never write into the same directory; the first complete one wins, a stale or
    failed entry is moved aside (kept for inspection, never deleted).
    """
    worker_sha = _sha256(request.root / CORD_WORKER)
    key = base_cache_key(request.base.hash, ids, decoding_of(request), worker_sha, image_digest(request.image) or "")
    caches = request.root / "outputs" / "eval-cache"
    named = f"{CORD_DATASET_ID}-{key}"  # "<catalog id>-<key>", like ocrbench-<key> (storage names it without guessing)
    cache_dir = caches / named
    for candidate in (cache_dir, caches / key):  # entries written before the prefix (bare key) are still read
        hit = _cache_hit(candidate, len(ids))
        if hit is not None:
            return hit
    tmp = cache_dir.with_name(f"{named}.tmp-{os.getpid()}-{secrets.token_hex(3)}")
    pred = _generate(request, request.base, ids, tmp, f"eval-base-{key[:12]}")
    if cache_dir.exists() and _cache_hit(cache_dir, len(ids)) is None:
        cache_dir.rename(cache_dir.with_name(f"{named}.stale-{time.strftime('%Y%m%d-%H%M%S')}"))
    try:
        tmp.rename(cache_dir)  # atomic on the same filesystem; fails if another node got there first
    except OSError:
        return pred  # someone else's complete entry is in place; ours stays valid in its tmp dir
    return cache_dir / pred.name


# ---------------------------------------------------------------------------------------------- score
PRED_MAX_BYTES = 64 * 1024 * 1024


def _load(pred: Path) -> dict[int, dict[str, Any]]:
    """Rows of a job-written pred.jsonl: never through a link, a regular file of bounded size (B1)."""
    fd = os.open(pred, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > PRED_MAX_BYTES:
            raise ValueError(f"{pred.name}: not a bounded regular file")
        text = fh.read(PRED_MAX_BYTES + 1).decode("utf-8")
    rows: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def _parsed(row: dict[str, Any]) -> Any:
    parsed = cord.prediction_of(row)
    return {} if parsed is None else parsed


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    scores = cord.score_rows(rows)
    return {
        "field_f1": round(scores.field_f1, 2),
        "json_exact": round(scores.json_exact, 2),
        "nted_accuracy": round(scores.nted_accuracy, 2),
        "parse_fail_rate": round(scores.parse_fail_rate, 2),
        "truncated_rate": round(100 * sum(bool(r["truncated"]) for r in rows) / len(rows), 2),
    }


def paired_f1(model: Sequence[dict[str, Any]], base: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """F1(model) - F1(base) on the same images, with a 95% percentile bootstrap CI over images (seeded)."""
    if [r["index"] for r in model] != [r["index"] for r in base]:
        raise ValueError("paired comparison needs the same images in the same order")
    a = [cord.field_f1_counts(_parsed(r), r["gt_parse"]) for r in model]
    b = [cord.field_f1_counts(_parsed(r), r["gt_parse"]) for r in base]
    rng = random.Random(BOOTSTRAP_SEED)  # noqa: S311 - statistics, seeded for reproducibility
    n = len(a)
    diffs = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100 * (cord.f1_from_counts([a[i] for i in idx]) - cord.f1_from_counts([b[i] for i in idx])))
    diffs.sort()
    return {
        "delta": round(100 * (cord.f1_from_counts(a) - cord.f1_from_counts(b)), 2),
        "ci95": [
            round(diffs[int(0.025 * BOOTSTRAP_RESAMPLES)], 2),
            round(diffs[int(0.975 * BOOTSTRAP_RESAMPLES) - 1], 2),
        ],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _records(m: dict[str, float], *, variant: str, n: int, common: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat benchmark records (frontend-metrics-contract), one per metric, values on 0-100."""
    higher = {
        "field_f1": True,
        "json_exact": True,
        "nted_accuracy": True,
        "parse_fail_rate": False,
        "truncated_rate": False,
    }
    return [
        {
            **common,
            "variant": variant,
            "metric": k,
            "value": v,
            "scale": "percent",
            "higher_is_better": higher[k],
            "n_samples": n,
        }
        for k, v in m.items()
    ]


# ------------------------------------------------------------------------------------------------ run
def run_eval_benchmarks(request: EvalRequest) -> EvalOutcome:
    check_support(request.node_id, request.params, request.entry, {}, root=request.root)  # backstop of validate
    if suite_of(request.params, request.entry) == "ocrbench":
        return run_ocrbench(request)
    ids = list(range(LEVEL_SAMPLES[request.params.level]))
    model_rows = _load(
        _generate(
            request, request.model, ids, request.run_dir / "jobs" / "model", f"{request.run_dir.name[:100]}-model"
        )
    )
    compare = request.params.compare_with == "base" and request.base.hash != request.model.hash
    base_rows = _load(_cached_base(request, ids)) if compare else None

    ordered = [model_rows[i] for i in ids]
    m = metrics(ordered)
    _, revision = _cord_snapshot(request.root)
    common = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": "cord_test",
        "split": "test",
        "dataset_revision": revision,
        "harness": "eval_cord_worker",
        "harness_version": _sha256(request.root / CORD_WORKER)[:16],
        "eval_path": "inprocess",
        "answer_extractor": "rule",
        "max_new_tokens": request.params.max_new_tokens,
        "thinking": False,
        "judge_model": None,
        "container_image": request.image,
        "container_digest": image_digest(request.image),
        "source": "measured",
    }
    records = _records(m, variant="model", n=len(ids), common=common)
    comparison = None
    if base_rows is not None:
        base_ordered = [base_rows[i] for i in ids]
        comparison = {"compare_to": "base", "metric": "field_f1", **paired_f1(ordered, base_ordered)}
        records += _records(metrics(base_ordered), variant="base", n=len(ids), common=common)
    gate = finetune_gate(
        [
            BenchmarkResult(
                benchmark="cord_test",
                n=len(ids),
                structured_output=True,
                truncated_pct=m["truncated_rate"],
                parse_fail_pct=m["parse_fail_rate"],
                delta=comparison["delta"] if comparison else None,
                delta_ci95=tuple(comparison["ci95"]) if comparison else None,
            )
        ]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "eval_report",
        "suite": "cord_test",
        "level": request.params.level,
        "model": {"path": str(request.model.path), "hash": request.model.hash},
        "base": {"path": str(request.base.path), "hash": request.base.hash},
        "decoding": decoding_of(request),
        "image_ids": ids,
        "records": records,
        "comparison": comparison,
        "gate": gate.to_json(),
        "samples_path": str(request.run_dir / "jobs" / "model" / "pred.jsonl"),
    }
    path = request.run_dir / "report.json"
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return EvalOutcome(path, hashlib.sha256(text.encode()).hexdigest(), gate.status.name)


# ------------------------------------------------------------------------------------- OCRBench (vLLM)
def _template_pixels(root: Path, entry: CatalogEntry) -> tuple[int, int]:
    """min/max pixels of the model's training template (processor), so eval sees images as training did."""
    path = configs_root(root) / TEMPLATE_DIR / f"{entry.id}.yaml"
    proc = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("processor", {}) if path.is_file() else {}
    return int(proc.get("min_pixels", 0) or 0), int(proc.get("max_pixels", 0) or 0)


def quantization_of(model: ModelRef) -> tuple[str, str] | None:
    """(precision, parent hash) of a quantized artifact ("<run_id>:<parent hash>:<format>"), else None."""
    parts = model.hash.split(":")
    if len(parts) == 3 and parts[2] in QUANT_PRECISION:
        return QUANT_PRECISION[parts[2]], parts[1]
    return None


def _ocrbench_data(root: Path) -> tuple[Path, str, str]:
    """(extracted copy, its manifest sha256, dataset revision[:8]); NET-002 when missing or not THE pinned copy."""
    entry = load_dataset_catalog().get(OCRBENCH_CATALOG_ID)
    params: dict[str, Any] = {"catalog_id": OCRBENCH_CATALOG_ID, "kind": "dataset"}
    if entry is None or entry.extracted_manifest_sha256 is None:  # fail closed: only THE pinned copy is scored
        raise EvalDataNotDownloadedError("OCRBench has no pinned copy", params={**params, "reason": "not_pinned"})
    params.update(repo=entry.repo, name=entry.display_name, license=entry.license)
    data = eval_data_dir(root, entry)
    if data.is_symlink() or not data.is_dir():
        raise EvalDataNotDownloadedError("OCRBench is not on this machine", params={**params, "reason": "not_cached"})
    try:
        return data, ocrbench.verify_manifest(data, entry.extracted_manifest_sha256), entry.revision[:8]
    except ocrbench.OcrBenchDataError as exc:
        raise EvalDataNotDownloadedError(str(exc), params={**params, "reason": "not_prepared"}) from exc


def _vllm_run(request: EvalRequest, model: ModelRef, rows: Sequence[dict[str, Any]], data: Path, *,
              quantized: bool, run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # fmt: skip
    """One vLLM batch (model or base) -> (per-question scores, engine facts); EvalJobFailed if it does not complete."""
    runner = request.vllm_batch
    if runner is None:
        raise EvalSuiteUnsupportedError("the vLLM runner is not available here", params={"node": request.node_id})
    lo, hi = _template_pixels(request.root, request.entry)
    res = runner(
        request.root, model.path, ocrbench.batch_requests(rows, data, ocrbench.MAX_TOKENS), who=request.who,
        tag=f"ocrbench-{run_dir.name}", quantization="modelopt" if quantized else None, min_pixels=lo,
        max_pixels=hi, enable_thinking=False, kv_cache_gb=ocrbench.KV_CACHE_GB, cancel=request.cancel,
        on_wait=request.on_wait, run_dir=run_dir,
    )  # fmt: skip
    if not res.ok:
        raise EvalJobFailed(read_manifest(res.run_dir), log_tail(res.run_dir))
    return ocrbench.score_batch(rows, res.rows()), res.engine()


def _ocrbench_base(request: EvalRequest, rows: Sequence[dict[str, Any]], data: Path, manifest_sha: str,
                   model_engine: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # fmt: skip
    """Base / bf16 parent scores from the shared cache, keyed on everything that determines them.

    A cached run whose engine differs from the model's (e.g. another vLLM image) is not reused.
    """
    lo, hi = _template_pixels(request.root, request.entry)
    material = json.dumps(
        {"base": request.base.hash, "data": manifest_sha, "rows": [r["row"] for r in rows], "pixels": [lo, hi],
         "max_tokens": ocrbench.MAX_TOKENS, "kv": ocrbench.KV_CACHE_GB, "thinking": False,
         "scorer": _sha256(Path(ocrbench.__file__))},
        sort_keys=True,
    )  # fmt: skip
    key = "ocrbench-" + hashlib.sha256(material.encode()).hexdigest()[:24]
    # The cached scores live in a HOST-owned file next to (never inside) the job's run dir: the job could plant a
    # link in its own dir (B1), and its outputs are only read through BatchResult.
    cache_file = request.root / "outputs" / "eval-cache" / f"{key}.json"
    cached = _read_cache(cache_file)
    if cached is not None and not ocrbench.engine_mismatch(cached[1], model_engine):
        return cached
    run_dir = cache_file.with_name(f"{key}.run-{os.getpid()}-{secrets.token_hex(3)}")
    scored, engine = _vllm_run(request, request.base, rows, data, quantized=False, run_dir=run_dir)
    _write_cache(cache_file, {"scored": scored, "engine": engine})
    return scored, engine


def _read_cache(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """(scores, engine) from a cache file, or None for anything missing, linked, oversized or malformed."""
    try:
        data = read_small_json(path, max_bytes=8 * 1024 * 1024)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    scored, engine = data.get("scored"), data.get("engine")
    if not isinstance(scored, list) or not isinstance(engine, dict):
        return None
    if not all(isinstance(r, dict) and r.get("score") in (0, 1) and isinstance(r.get("row"), int) for r in scored):
        return None
    return scored, engine


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write of a host-owned cache file: a fresh name (O_EXCL | O_NOFOLLOW), then rename over the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)


def run_ocrbench(request: EvalRequest) -> EvalOutcome:
    """OCRBench on the vLLM batch: the model and its base (bf16 parent when quantized), paired by question.

    Quantized -> quant_gate; else finetune_gate with OCRBench as a general benchmark (forgetting check).
    """
    data, manifest_sha, revision = _ocrbench_data(request.root)
    rows = ocrbench.questions(data, subset=request.params.level == "quick")
    quant = quantization_of(request.model)
    if quant is not None and quant[1] != request.base.hash:
        raise EvalSuiteUnsupportedError(
            "comparing a quantized fine-tuned model with its parent is not available yet",
            params={"node": request.node_id, "node_type": "eval.benchmarks"},
        )
    scored, engine = _vllm_run(request, request.model, rows, data, quantized=quant is not None,
                               run_dir=request.run_dir / "jobs" / "model")  # fmt: skip
    compare = quant is not None or (request.params.compare_with == "base" and request.base.hash != request.model.hash)
    base_scored, base_engine, comparison = None, None, None
    if compare:
        base_scored, base_engine = _ocrbench_base(request, rows, data, manifest_sha, engine)
        if mismatch := ocrbench.engine_mismatch(engine, base_engine):
            raise EvalSuiteUnsupportedError(f"base and model ran on different engines: {mismatch}",
                                            params={"node": request.node_id})  # fmt: skip
        pa = ocrbench.paired_accuracy([s["score"] for s in scored], [s["score"] for s in base_scored])
        comparison = {"compare_to": "parent" if quant else "base", "metric": "accuracy", **pa}
    summ = ocrbench.summary(scored)
    common = {
        "schema_version": REPORT_SCHEMA_VERSION, "benchmark": "ocrbench",
        "split": "regression_500" if request.params.level == "quick" else "test", "dataset_revision": revision,
        "data_manifest_sha256": manifest_sha, "harness": "vllm_batch", "eval_path": "vllm",
        "answer_extractor": "substring", "max_new_tokens": ocrbench.MAX_TOKENS, "thinking": False,
        "judge_model": None, "container_image": engine.get("image"), "container_digest": engine.get("container_digest"),
        "source": "measured", "citation": ocrbench.CITATION,
    }  # fmt: skip

    def recs(s: dict[str, float], variant: str) -> list[dict[str, Any]]:
        higher = {"accuracy": True, "truncated_rate": False, "error_rate": False}
        return [{**common, "variant": variant, "metric": k, "value": s[k], "scale": "percent",
                 "higher_is_better": higher[k], "n_samples": len(rows)} for k in higher]  # fmt: skip

    records = recs(summ, "model")
    if base_scored is not None:
        records += recs(ocrbench.summary(base_scored), "parent" if quant else "base")
    result = BenchmarkResult(
        benchmark="ocrbench", n=len(rows), truncated_pct=summ["truncated_rate"],
        delta=comparison["delta"] if comparison else None,
        delta_ci95=tuple(comparison["ci95"]) if comparison else None, general=quant is None,
    )  # fmt: skip
    gate = quant_gate([result], quant[0]) if quant else finetune_gate([result])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": "eval_report", "suite": "ocrbench",
        "level": request.params.level, "model": {"path": str(request.model.path), "hash": request.model.hash},
        "base": {"path": str(request.base.path), "hash": request.base.hash},
        "quantization": None if quant is None else {"precision": quant[0], "parent": quant[1]},
        "decoding": {"sampling": engine.get("sampling"), "max_new_tokens": ocrbench.MAX_TOKENS, "thinking": False},
        "engine": {"model": engine, "base": base_engine}, "records": records, "comparison": comparison,
        "gate": gate.to_json(), "rows": [s["row"] for s in scored],
        "samples_path": str(request.run_dir / "jobs" / "model" / "out.jsonl"),
        "publish": "aggregate scores only (OCRBench eval_only, security 2026-10-13)",
    }  # fmt: skip
    path = request.run_dir / "report.json"
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return EvalOutcome(path, hashlib.sha256(text.encode()).hexdigest(), gate.status.name)
