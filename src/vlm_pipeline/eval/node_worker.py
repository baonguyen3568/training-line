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
from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    MemoryPreemptedBeforeStartError,
    image_digest,
    run_job,
)
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, read_manifest
from vlm_pipeline.eval import cord
from vlm_pipeline.eval.config import EvalParams
from vlm_pipeline.eval.gate import BenchmarkResult, finetune_gate

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


class EvalSuiteUnsupportedError(VlmPipelineError):
    """The requested suite / model has no evaluation engine yet (only CORD on Nemotron-Parse so far)."""

    code: ClassVar[str] = "CFG-011"


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
    del inputs  # cord_test always reads CORD-v2 test (heldout); a wired dataset matters once suite my_val exists
    if model is None:  # no model wired: CFG-003 from validate, nothing to say about support
        return

    def refuse(why: str) -> EvalSuiteUnsupportedError:
        return EvalSuiteUnsupportedError(why, params={"node": node_id, "node_type": "eval.benchmarks"})

    if params.suite not in ("auto", "cord_test") or model.loss != "nemotron_parse":
        raise refuse(f"suite {params.suite!r} is not available for {model.id} yet")
    if params.thinking:
        raise refuse("thinking mode does not apply to Nemotron-Parse")
    unsupported = set(template_overrides(root or project_root(), model)) - SUPPORTED_OVERRIDES
    if unsupported:  # never evaluate with a decoding other than the one "Chạy thử" uses
        raise refuse(f"decode_overrides {sorted(unsupported)} not supported by the CORD eval worker")


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
    path = root / TEMPLATE_DIR / f"{entry.id}.yaml"
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
            ContainerJob(name=manifest.run_id, image=request.image, argv=argv),
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
        raise EvalJobFailed(final, _tail(run_dir / "logs" / "stdout.log"))
    return run_dir / "pred.jsonl"


def _cache_hit(cache_dir: Path, n: int) -> Path | None:
    pred = cache_dir / "pred.jsonl"
    complete = pred.is_file() and (cache_dir / "run.json").is_file()
    if complete and read_manifest(cache_dir).status == RunStatus.COMPLETED and len(_load(pred)) == n:
        return pred
    return None


def _cached_base(request: EvalRequest, ids: Sequence[int]) -> Path:
    """Base predictions from the shared cache, or generated into a private dir then renamed into place atomically.

    Two eval nodes on the same base never write into the same directory; the first complete one wins, a stale or
    failed entry is moved aside (kept for inspection, never deleted).
    """
    worker_sha = _sha256(request.root / CORD_WORKER)
    key = base_cache_key(request.base.hash, ids, decoding_of(request), worker_sha, image_digest(request.image) or "")
    cache_dir = request.root / "outputs" / "eval-cache" / key
    hit = _cache_hit(cache_dir, len(ids))
    if hit is not None:
        return hit
    tmp = cache_dir.with_name(f"{key}.tmp-{os.getpid()}-{secrets.token_hex(3)}")
    pred = _generate(request, request.base, ids, tmp, f"eval-base-{key[:12]}")
    if cache_dir.exists() and _cache_hit(cache_dir, len(ids)) is None:
        cache_dir.rename(cache_dir.with_name(f"{key}.stale-{time.strftime('%Y%m%d-%H%M%S')}"))
    try:
        tmp.rename(cache_dir)  # atomic on the same filesystem; fails if another node got there first
    except OSError:
        return pred  # someone else's complete entry is in place; ours stays valid in its tmp dir
    return cache_dir / pred.name


# ---------------------------------------------------------------------------------------------- score
def _load(pred: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in pred.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def _parsed(row: dict[str, Any]) -> Any:
    parsed = cord.token2json(row["sequence"])
    return {} if cord.is_parse_failure(parsed) else parsed


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    scores = cord.score_sequences([r["sequence"] for r in rows], [r["gt_parse"] for r in rows])
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


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""
