"""Real node worker for the app (``--worker container``): host source nodes + AutoModel fine-tune jobs.

Implements ``graph.worker.Worker``. Heavy nodes go through ``common.jobrun.run_job`` so every job gets the preflight,
the GPU lock, the host memory watchdog and run.json. Errors are raised, never classified here (the executor maps
them to catalog codes). Nothing is downloaded: models/datasets must already be in the project HF cache (offline jobs).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.app.bench_worker import run_bench
from vlm_pipeline.app.data_workers import run_dedupe, run_filter
from vlm_pipeline.app.export_worker import run_export
from vlm_pipeline.app.qat_worker import run_qat
from vlm_pipeline.app.quantize_worker import run_quantize
from vlm_pipeline.app.trial_worker import run_trial
from vlm_pipeline.common.catalog import (
    LICENSE_UNKNOWN,
    Catalog,
    CatalogEntry,
    load_catalog,
    load_dataset_catalog,
    model_entry,
    normalize_license,
)
from vlm_pipeline.common.dataset_ref import DatasetRef, read_dataset_ref, write_dataset_ref
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.heldout_data import HeldoutDataInTrainingError, is_heldout_folder
from vlm_pipeline.common.host_paths import hub_repo_dir
from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    MemoryPreemptedBeforeStartError,
    run_job,
)
from vlm_pipeline.common.local_cache import (
    PREPARED_DIRS,
    dataset_missing,
    hub_snapshot,
    model_missing,
    prepared_model_dir,
)
from vlm_pipeline.common.model_ref import PickedModel, is_model_ref, resolve_model_ref, write_model_ref
from vlm_pipeline.common.run_log import log_tail
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, automodel_to_metrics, read_manifest
from vlm_pipeline.download.resolver import block_prefixes
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import (
    NOT_RESUMABLE_MARKER,
    ArtifactRef,
    NodeJob,
    NodeResult,
    SupportQuery,
    checkpoint_dirs,
    port_of,
)
from vlm_pipeline.nodes.source import picked_model_id
from vlm_pipeline.nodes.spec import NodeSpec, is_compatible, model_kind
from vlm_pipeline.nodes.train import TRIAL_CAPABLE
from vlm_pipeline.train.advanced import AdvancedNotForModelError, keys_not_for
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams
from vlm_pipeline.train.recipe import Blocks, build_recipe, qlora_models, render_recipe, templates_digest

logger = logging.getLogger(__name__)
# Models whose AutoModel checkpoints cannot be resumed: a resume starts over instead of failing on every retry.
# Empty since vlm-automodel:26.08-6 (patch 0005, BUG-003 closed: Parse resumes, also from 26.08-5 checkpoints).
# Run dirs marked .not-resumable earlier stay so (graph.worker.NOT_RESUMABLE_MARKER).
RESUME_UNSUPPORTED: frozenset[str] = frozenset()


class TrainBlocksNotForModelError(VlmPipelineError):
    """train_blocks needs module prefixes (families.yaml / catalog) this model does not have (CFG-005)."""

    code: ClassVar[str] = "CFG-005"


class QloraNotForModelError(VlmPipelineError):
    """QLoRA (quantize_base=nf4) on a model whose template does not list the modules that stay bf16 (CFG-005)."""

    code: ClassVar[str] = "CFG-005"


class NotCachedError(VlmPipelineError):
    """A model or dataset is not in the project cache; jobs run offline, so it must be downloaded first."""

    code: ClassVar[str] = "NET-002"  # "not on this machine yet", not a network / sign-in failure (NET-001)


class UserDatasetNotAddedError(VlmPipelineError):
    """The workflow names a user dataset this machine never added (DATA-013: offer to add it)."""

    code: ClassVar[str] = "DATA-013"


class LocalDatasetInvalidError(VlmPipelineError):
    """The user's dataset folder has lines/images that cannot be used (all problems listed in problems.json)."""

    code: ClassVar[str] = "DATA-006"


class TrainModeUnsupportedError(VlmPipelineError):
    """The fine-tune node is not the one the catalog vetted for this model (raised before the GPU lock)."""

    code: ClassVar[str] = "CFG-010"


class NodeNotRunnableError(VlmPipelineError):
    """This worker has no implementation for the node type yet (validate reports it first via ``supports``)."""

    code: ClassVar[str] = "CFG-011"


class UnknownModelError(VlmPipelineError):
    """A model reference that is not a catalog model (validated params make this an internal error)."""

    code: ClassVar[str] = "SYS-000"


def caller_id() -> str:
    """``script:<program>/<pid>`` for a worker built outside the app, so the lock tells a script run from an app run."""
    program = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "python"
    return f"script:{program[:40]}/{os.getpid()}"


class RealWorker:
    def __init__(
        self,
        root: Path,
        *,
        catalog: Catalog | None = None,
        image: str = DEFAULT_TRAIN_IMAGE,
        who: str | None = None,
    ) -> None:
        self.root = root.resolve()
        self.catalog = catalog or load_catalog()
        self.image = image
        self.who = who or caller_id()  # GPU_LOCK "who": the app passes "vlm-app"; scripts say who they are

    def identity(self, spec: NodeSpec) -> str:
        """Part of the executor cache key: container nodes are keyed by the exact image (content id) and recipes."""
        if spec.runner != "container":
            return "host/real-1"
        return f"container/{self.image}@{self._image_id()}+recipe:{self._recipes_id()}"

    @functools.cache  # noqa: B019 - read once per app process like the image id; a template edit needs a restart
    def _recipes_id(self) -> str:
        return templates_digest(self.root)

    @functools.cache  # noqa: B019 - one worker per app process; the image id does not change while it runs
    def _image_id(self) -> str:
        from vlm_pipeline.common.jobrun import image_digest

        return image_digest(self.image) or "unknown"

    # ------------------------------------------------------------------------------------------ helpers
    def container_path(self, host_path: Path) -> str:
        return f"{CONTAINER_ROOT}/{host_path.resolve().relative_to(self.root).as_posix()}"

    def host_path(self, path: str) -> Path:
        """Artifact paths are host paths; container paths (older entries) are mapped back."""
        if path.startswith(CONTAINER_ROOT + "/"):
            return self.root / path.removeprefix(CONTAINER_ROOT + "/")
        return Path(path)

    def _entry_for(self, ref: ArtifactRef) -> CatalogEntry:
        for entry in self.catalog.models:
            if ref.hash == f"{entry.repo}@{entry.revision}":
                return entry
        from vlm_pipeline.app.user_assets import list_all  # ADR-021: a user-added model is a base model too

        for record in list_all(self.root):
            if ref.hash == f"{record.entry.repo}@{record.entry.revision}":
                return record.entry
        raise UnknownModelError("base model is not a catalog model", params={"hash": ref.hash})

    # ------------------------------------------------------------------------------------------ nodes
    @functools.cached_property
    def _handlers(self) -> dict[str, Callable[[NodeJob], NodeResult]]:
        """Node types this worker really runs; every other type is refused at validate time (CFG-011)."""
        return {
            "model.hf_pick": self._model_pick,
            "data.hf_load": self._hub_dataset,
            "data.split": self._split,
            "data.local_folder": self._local_folder,
            "train.lora_vlm": self._train,
            "train.sft_vlm": self._train,
            "train.merge_lora": self._merge_lora,
            "eval.benchmarks": self._eval,
            "bench.perf": functools.partial(run_bench, self),  # B-6 (app/bench_worker.py, bao-tester)
            "export.hf_dir": functools.partial(run_export, self),  # B-2: native models (ADR-012)
            "quantize.modelopt_ptq": functools.partial(run_quantize, self),  # B-4 (app/quantize_worker.py, research)
            "quantize.qat": functools.partial(run_qat, self),  # ADR-022 #10 (app/qat_worker.py, fixcode)
            "serve.vllm": functools.partial(run_trial, self),  # B-7 "Chạy thử" (app/trial_worker.py)
            "data.caption_vlm": self._caption,  # app/caption_worker.py (bao-fixcode)
            "data.filter": functools.partial(run_filter, self),  # app/data_workers.py (local datasets, v1)
            "data.dedupe": functools.partial(run_dedupe, self),
        }

    def _caption(self, job: NodeJob) -> NodeResult:
        from vlm_pipeline.app.caption_worker import run_caption  # imports this module: loaded on first use

        return run_caption(self, job)

    def supports(self, spec: NodeSpec) -> bool:
        return spec.type_id in self._handlers

    def run(self, job: NodeJob) -> NodeResult:
        handler = self._handlers.get(job.spec.type_id)
        if handler is None:
            raise NodeNotRunnableError(
                f"no worker for {job.spec.type_id}", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        self.check_support(self.support_query(job))  # the SAME check graph validate ran on the canvas
        if job.trial_max_steps and job.spec.type_id not in TRIAL_CAPABLE:  # never a full run in a trial's place
            raise NodeNotRunnableError(
                f"no quick trial for {job.spec.type_id}", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        return handler(job)

    # ------------------------------------------------------------------------------------------ support
    def support_query(self, job: NodeJob) -> SupportQuery:
        """The run-time SupportQuery of ``job`` (same fields graph validate derives from the canvas)."""
        model: CatalogEntry | None = None
        inputs: dict[str, str] = {}
        for key, ref in job.inputs.items():
            port = port_of(key)  # a many-port's extra wires (heldout.1...) have the same kind as the port
            port_spec = job.spec.inputs.get(port)
            if port_spec is not None and is_compatible(port_spec.type, "Model"):
                inputs[port] = model_kind(ref.type)  # the same kind the canvas derives from the output port
                if model is None:
                    try:
                        model = self.base_of(ref)
                    except VlmPipelineError:
                        model = None  # untraceable: the handler reports it (not a support question)
            elif ref.type == "Dataset":
                data = self.dataset_input(ref)
                inputs[port] = "dataset:local" if data.records else ("dataset:hub" if data.repo else "other")
            else:
                inputs[port] = "other"
        return SupportQuery(job.node_id, job.spec, job.params, model, inputs)

    def check_support(self, query: SupportQuery) -> None:
        """Raise the catalog error for a node this worker will not run with this model / params / inputs.

        Pure (repo config only, never user or run-time files) so graph validate can call it: a clean canvas means
        the worker does not refuse later for a support reason. Each node type's rule lives with its owner.
        """
        check = _SUPPORT_CHECKS.get(query.spec.type_id)
        if check is not None:
            check(query)

    def _model_pick(self, job: NodeJob) -> NodeResult:
        model_id = picked_model_id(job.params)
        entry = model_entry(model_id) if model_id.startswith("user--") else self.catalog.get(model_id)
        if entry is None:
            raise UnknownModelError("unknown catalog model", params={"model": model_id})
        path = self._catalog_model_path(entry).resolve()
        # Outputs live in the run dir (executor check_output_paths): a descriptor, the weights stay in the cache.
        picked = PickedModel(repo=entry.repo, revision=entry.revision, path=path.relative_to(self.root).as_posix())
        descriptor = write_model_ref(job.run_dir.resolve(), picked)
        ref = ArtifactRef(type="Model", path=str(descriptor), hash=f"{entry.repo}@{entry.revision}")
        return NodeResult(outputs={"model": ref})

    def _train_inputs(
        self, job: NodeJob, entry: CatalogEntry, base: ArtifactRef, train: DatasetRef, val: DatasetRef | None
    ) -> tuple[Path, ...]:
        """Read-only inputs of a train job: base model, data (+ val), and the resume checkpoint step (C3)."""
        inputs = [self.mount_input(self.model_dir(base)), self.mount_input(train.path)]
        if val is not None:
            inputs.append(self.mount_input(val.path))
        if job.resume_from is not None and entry.id not in RESUME_UNSUPPORTED:
            inputs.append(job.resume_from)  # exactly that checkpoint step
        return tuple(inputs)

    def mount_input(self, path: str | Path) -> Path:
        """The host dir/file a narrowed job mounts read-only to read ``path`` (P017 step 6).

        An HF hub path mounts its whole repo dir (snapshot links point into ``blobs/``).
        """
        return hub_repo_dir(self.root, self.no_latest(self.host_path(str(path))))

    def no_latest(self, path: Path) -> Path:
        """``path`` with ``checkpoints/LATEST`` (a symlink the job wrote) replaced by the newest real checkpoint dir.

        Chosen by the host from dir names (P017 C2): a job path or mount source never goes through a job's symlink.
        """
        if "LATEST" not in path.parts:
            return path
        i = path.parts.index("LATEST")
        ckpts = checkpoint_dirs(Path(*path.parts[: i - 1]))
        if not ckpts:
            return path
        newest: Path = ckpts[-1][1]
        return newest.joinpath(*path.parts[i + 1 :])

    def model_dir(self, ref: ArtifactRef) -> Path:
        """Host folder of a Model artifact: a picked catalog model's cache folder, or a trained model's run folder."""
        path = self.host_path(ref.path)
        # A plain folder is a trained model, or a pick cached before model.json existed (legacy, read-only: a cache
        # hit reuses the entry without check_output_paths, which runs only on fresh results).
        return resolve_model_ref(path, self.root) if is_model_ref(path) else self.no_latest(path)

    def _catalog_model_path(self, entry: CatalogEntry) -> Path:
        """Host dir of a catalog model: the pinned prepared copy (remote code) or the hub snapshot."""
        if (missing := model_missing(self.root, entry)) is not None:  # validate warned already (NET-002)
            raise NotCachedError(f"{entry.repo}@{entry.revision}: {missing['reason']}", params=missing)
        prepared = prepared_model_dir(self.root, entry.id) if entry.remote_code == "pinned_prepared" else None
        return prepared or hub_snapshot(self.root, "model", entry.repo, entry.revision)

    def base_of(self, model: ArtifactRef) -> CatalogEntry:
        """Catalog model a Model/Checkpoint artifact comes from: itself, or the base recorded in its run.json."""
        if "@" in model.hash.split(":", 1)[0]:  # a picked catalog model (model.json): never reaches the run.json walk
            return self._entry_for(model)
        path = self.host_path(model.path).resolve()
        outputs = self.root / "outputs"
        within = [d for d in (path, *path.parents) if d.is_relative_to(outputs) and d != outputs]  # never above
        run_dir = next((d for d in within if (d / "run.json").is_file()), None)
        if run_dir is None:
            raise UnknownModelError("model has no run record", params={"hash": model.hash})
        manifest = read_manifest(run_dir)
        ref = ArtifactRef("Model", "", f"{manifest.base_model}@{manifest.base_model_revision}")
        return self._entry_for(ref)

    def _eval(self, job: NodeJob) -> NodeResult:
        """Benchmark node (B-5): bao-tester's engine (eval/node_worker.py) on the model and its catalog base."""
        from vlm_pipeline.eval.node_worker import EvalJobFailed, EvalRequest, ModelRef, run_eval_benchmarks
        from vlm_pipeline.serve.vllm_runner import run_vllm_batch

        model = job.inputs["model"]
        entry = self.base_of(model)
        base_path = self._catalog_model_path(entry)
        request = EvalRequest(
            root=self.root, node_id=job.node_id, params=job.params,  # type: ignore[arg-type]
            model=ModelRef(self.model_dir(model), model.hash),
            base=ModelRef(base_path.resolve(), f"{entry.repo}@{entry.revision}"),
            entry=entry, run_dir=job.run_dir.resolve(), image=self.image, who=self.who,
            cancel=job.cancel, on_wait=job.on_wait,
            vllm_batch=run_vllm_batch,  # suite ocrbench (eval may not import serve: injected here)
        )  # fmt: skip
        try:
            outcome = run_eval_benchmarks(request)
        except EvalJobFailed as exc:
            raise ContainerExit(exc.exit_code, oom_killed=bool(exc.final.oom_killed),
                                cancelled_by_executor=exc.final.status == RunStatus.CANCELLED,
                                log_tail=exc.log_tail) from exc  # fmt: skip
        report = ArtifactRef(type="EvalReport", path=str(outcome.report_path), hash=outcome.report_hash)
        return NodeResult(outputs={"report": report})

    def _hub_dataset(self, job: NodeJob) -> NodeResult:
        if user_id := str(getattr(job.params, "user_dataset", "") or ""):
            return self._user_hub_dataset(job, user_id)
        repo = str(getattr(job.params, "repo"))  # noqa: B009
        entry = next((d for d in load_dataset_catalog().datasets if d.repo == repo), None)
        if entry is None:
            raise NotCachedError("dataset not in the catalog", params={"repo": repo, "reason": "not_pinned"})
        revision = entry.revision
        if (missing := dataset_missing(self.root, entry)) is not None:
            raise NotCachedError(f"{repo}@{revision} not in cache", params=missing)
        path = hub_snapshot(self.root, "dataset", repo, revision)
        limit = int(getattr(job.params, "max_samples", 0) or 0)
        # The artifact is a small descriptor (snapshot + split + sample limit), so "quick run" sizes cost no copy.
        ref = DatasetRef(repo=repo, revision=revision, path=self.container_path(path), limit=limit)
        return NodeResult(outputs={"dataset": self.dataset_artifact(job.run_dir / "dataset.json", ref)})

    def dataset_artifact(self, path: Path, ref: DatasetRef) -> ArtifactRef:
        """Write ``ref`` as a node's Dataset output (also used by the data / caption worker modules)."""
        write_dataset_ref(path, ref)
        return ArtifactRef(type="Dataset", path=str(path.resolve()), hash=ref.hash)

    def dataset_input(self, ref: ArtifactRef) -> DatasetRef:
        """The descriptor behind a Dataset artifact (a plain folder reads as all of it, train split)."""
        if ref.path.endswith(".json"):
            return read_dataset_ref(self.host_path(ref.path))
        return DatasetRef(path=self.container_path(self.host_path(ref.path)))

    def _local_folder(self, job: NodeJob) -> NodeResult:
        """User folder of images + labels.jsonl -> checked, normalized records (B-3b). Host, CPU only."""
        folder = (self.root / str(getattr(job.params, "path"))).resolve()  # noqa: B009 - LocalPath (SEC-001)
        if not folder.is_relative_to(self.root) or not folder.is_dir():
            missing = {"node": job.node_id, "problem_count": 1, "first_line": 0, "first_reason": "labels_missing"}
            raise LocalDatasetInvalidError("dataset folder missing", params=missing)
        return self._folder_dataset(job, folder)

    def _user_hub_dataset(self, job: NodeJob, dataset_id: str) -> NodeResult:
        """A user-added HF split (ADR-021 H5): its converted folder, checked exactly like a local folder (DATA-V)."""
        from vlm_pipeline.app import user_datasets

        record = user_datasets.load(self.root, dataset_id)
        folder = user_datasets.data_folder(self.root, dataset_id)
        if record is None:
            raise UserDatasetNotAddedError("dataset not added here", params={
                "node": job.node_id, "repo": str(getattr(job.params, "user_repo", "")),
                "revision": str(getattr(job.params, "user_revision", ""))})  # fmt: skip
        if folder.is_symlink() or not (folder / "labels.jsonl").is_file():
            params = {"name": record.to_api()["name"], "kind": "dataset", "catalog_id": record.id,
                      "size_gb": record.to_api()["size_gb"], "reason": "not_cached"}  # fmt: skip
            raise NotCachedError("user dataset not downloaded", params=params)
        return self._folder_dataset(job, folder.resolve(), origins=(dataset_id,))  # its licence follows (H5d)

    def _folder_dataset(self, job: NodeJob, folder: Path, origins: tuple[str, ...] = ()) -> NodeResult:
        """A folder of images + labels.jsonl -> checked, normalized records, the local dataset artifact."""
        from vlm_pipeline.common.local_records import write_records
        from vlm_pipeline.data.local_folder import scan_folder

        params = job.params
        scan = scan_folder(folder, max_samples=int(getattr(params, "max_samples", 0) or 0),
                           max_image_side_px=int(getattr(params, "max_image_side_px", 4096)))  # fmt: skip
        job.run_dir.mkdir(parents=True, exist_ok=True)
        if scan.problem_count:
            (job.run_dir / "problems.json").write_text(json.dumps(scan.problems, ensure_ascii=False, indent=1))
            first = scan.problems[0]
            raise LocalDatasetInvalidError(
                f"{scan.problem_count} unusable line(s) in labels.jsonl",
                params={"node": job.node_id, "problem_count": scan.problem_count, "first_line": first["line"],
                        "first_reason": first["reason"]},
            )  # fmt: skip
        records = job.run_dir / "records.jsonl"
        sha = write_records(records, scan.records)
        ref = DatasetRef(
            path=self.container_path(folder), split="all", records=self.container_path(records), records_sha256=sha,
            seed=int(getattr(params, "seed", 0)), origins=origins,
        )  # fmt: skip
        return NodeResult(outputs={"dataset": self.dataset_artifact(job.run_dir / "dataset.json", ref)})

    def _split(self, job: NodeJob) -> NodeResult:
        """Train / val of a dataset; held-out test splits never appear.

        Hub datasets with an official validation split use it (results stay comparable with published numbers);
        ``val_fraction`` applies to data without one.
        """
        src = self.dataset_input(job.inputs["dataset"])
        if src.records:  # local data: stable hash split, recorded so run.json can reproduce it
            fraction, seed = float(getattr(job.params, "val_fraction")), int(getattr(job.params, "seed"))  # noqa: B009
            rule = {"val_fraction": fraction, "seed": seed}
            return NodeResult(
                outputs={
                    "train": self.dataset_artifact(job.run_dir / "train.json",
                                                    src.model_copy(update={"split": "train", **rule})),
                    "val": self.dataset_artifact(job.run_dir / "val.json",
                                                  src.model_copy(update={"split": "validation", "limit": 0, **rule})),
                }
            )  # fmt: skip
        entry = next((d for d in load_dataset_catalog().datasets if src.repo and d.repo == src.repo), None)
        if entry is None or "validation" not in entry.splits or "validation" in entry.heldout_splits:
            raise NodeNotRunnableError(  # local folders: fraction split comes with data.local_folder (B-3)
                "split by fraction not implemented yet", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        train = src.model_copy(update={"split": "train"})
        val = src.model_copy(update={"split": "validation", "limit": 0})  # quick-run size limits training only
        return NodeResult(
            outputs={
                "train": self.dataset_artifact(job.run_dir / "train.json", train),
                "val": self.dataset_artifact(job.run_dir / "val.json", val),
            }
        )

    def _train(self, job: NodeJob) -> NodeResult:
        params = job.params
        if not isinstance(params, LoraParams | FullFinetuneParams):
            raise NodeNotRunnableError(
                "unexpected params model", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        base, data = job.inputs["base_model"], job.inputs["train_data"]
        entry = self._entry_for(base)
        class_token = None
        if entry.loss == "nemotron_parse":
            from vlm_pipeline.train.recipe_checks import resolve_parse_class_token_start_idx

            class_token = resolve_parse_class_token_start_idx(self.model_dir(base))
        run_dir = job.run_dir.resolve()
        if entry.id in RESUME_UNSUPPORTED:
            _set_aside_checkpoints(run_dir, entry.id)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        if entry.id in RESUME_UNSUPPORTED:
            (run_dir / NOT_RESUMABLE_MARKER).touch()  # executor: no "resume" offer, warn on the next run
        train = self.dataset_input(data)
        val = self.dataset_input(job.inputs["val_data"]) if "val_data" in job.inputs else None
        for used in (train, val):  # DATA-007 at run time too: a canvas edited outside the app never trains on test data
            if used is not None and used.repo is None and is_heldout_folder(self.root, self.host_path(used.path)):
                raise HeldoutDataInTrainingError(
                    "train data under data/eval", params={"node": job.node_id, "target_node": job.node_id}
                )
        from vlm_pipeline.app.user_assets import training_template

        template = training_template(self.root, entry)  # a user model's: only with its recorded sha256
        if val is not None and "validation_dataset" not in template:
            logger.warning("%s: template has no validation_dataset; the wired val_data is not used", entry.id)
        source = next((d for d in load_dataset_catalog().datasets if train.repo and d.repo == train.repo), None)
        no_val_split = source is None or "validation" not in source.splits or "validation" in source.heldout_splits
        train_meta, val_meta, train_prompt = self._local_meta(run_dir, entry, train, val, node_id=job.node_id)
        recipe = build_recipe(
            template,
            params,
            model_dir=self.container_path(self.model_dir(base)),
            dataset_dir=train.path,
            train_limit=train.limit or None,
            train_split=train.split,
            val_dataset_dir=val.path if val else None,
            val_split=val.split if val else None,
            drop_validation=val is None and no_val_split,
            train_meta=train_meta,
            val_meta=val_meta,
            run_dir=self.container_path(run_dir),
            class_token_start_idx=class_token,
            model_id=entry.id,
            blocks=_model_blocks(entry),  # P023 §1: train_blocks / vision_lr_scale act on these modules
            trial_steps=job.trial_max_steps,  # a quick trial run (MODEL-005): a run option, never a param
        )
        if job.resume_from is not None and entry.id not in RESUME_UNSUPPORTED:
            # P017: the earlier attempt's checkpoint (chosen and checked by the executor, C1-C3) is read through the
            # project mount; weights + optimizer + dataloader state continue, new checkpoints go to this attempt.
            recipe["checkpoint"]["restore_from"] = self.container_path(job.resume_from)  # build_recipe made it
        text = render_recipe(recipe)
        (run_dir / "config.yaml").write_text(text, encoding="utf-8")
        config_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        run_id = run_dir.name[:120]  # executor run dir = <node>-<cache key prefix>[-a<N>]: one per job attempt
        manifest = RunManifest(
            run_id=run_id,
            name=f"{job.spec.type_id} {entry.display_name}",
            stage="train",
            task="lora" if isinstance(params, LoraParams) else "sft",
            base_model=entry.repo,
            base_model_revision=entry.revision,
            resumed_from=(
                {"run": job.resume_from.parent.parent.name, "checkpoint": job.resume_from.name}
                if job.resume_from is not None and entry.id not in RESUME_UNSUPPORTED
                else None
            ),
            dataset=data.hash,
            config_path=self.container_path(run_dir / "config.yaml"),
            config_hash=config_hash,
            seed=params.seed,
            container_image=self.image,
            tags=["trial"] if job.trial_max_steps else [],  # the Runs page: "Chạy thử", never a real training
            train_prompt=train_prompt,
        )
        container = ContainerJob(
            name=run_id,
            image=self.image,
            argv=("automodel", self.container_path(run_dir / "config.yaml"), "--nproc-per-node", "1"),
            workdir="/opt/Automodel",
            env=_trainable_env(entry, params),
            narrow=True,
            rw_dirs=(run_dir,),
            ro_inputs=self._train_inputs(job, entry, base, train, val),
        )
        try:
            final = run_job(
                container,
                manifest,
                run_dir,
                project_root=self.root,
                who=self.who,
                est_peak_gb=entry.ram_gb_train,
                ckpt_gb=entry.disk_gb,
                cancel=job.cancel,
                on_wait=job.on_wait,
                require_approved_image=True,  # user jobs never run on a PROBE image
            )
        except MemoryPreemptedBeforeStartError as exc:
            raise InsufficientMemoryError(str(exc), params={"need_gb": entry.ram_gb_train}) from exc
        self._write_metrics(run_dir)
        if final.status != RunStatus.COMPLETED:
            raise ContainerExit(
                75 if final.status == RunStatus.INTERRUPTED else (final.exit_code or 1),
                oom_killed=bool(final.oom_killed),
                cancelled_by_executor=final.status == RunStatus.CANCELLED,
                log_tail=log_tail(run_dir),
            )
        port, out_type = next(iter(job.spec.outputs.items()))
        out = run_dir / "checkpoints" / "LATEST" / "model"
        if (out / "consolidated").is_dir():  # full fine-tune: HF safetensors loadable by downstream nodes
            out = out / "consolidated"
            from vlm_pipeline.train.ckpt_complete import complete_from_base

            # BUG-002: consolidation drops buffers (RADIO input conditioner) -> restore them from the base model.
            complete_from_base(self.model_dir(base), out)
        ref = ArtifactRef(type=out_type.type, path=str(out), hash=f"{run_id}:{config_hash}")
        return NodeResult(outputs={port: ref})

    def _local_meta(
        self, run_dir: Path, entry: CatalogEntry, train: DatasetRef, val: DatasetRef | None, *, node_id: str
    ) -> tuple[str | None, str | None, str | None]:
        """Local data -> model-specific make_meta_dataset files in the train run dir (container paths), else None.

        Third: the one question every training sample is asked (H-prompt), None when they differ.
        """
        if not train.records:
            return None, None, None
        from vlm_pipeline.common.local_records import (
            Family,
            image_path,
            read_records,
            select,
            single_prompt,
            write_meta_dataset,
        )

        family: Family = "nemotron_parse" if entry.loss == "nemotron_parse" else "chat"
        out = run_dir / "data"

        prompts: dict[str, str | None] = {}

        def meta(ref: DatasetRef, name: str) -> str:
            records = list(read_records(self.host_path(str(ref.records))))
            chosen = select(records, part=ref.split, seed=ref.seed or 0, val_fraction=ref.val_fraction or 0.0,
                            limit=ref.limit)  # fmt: skip
            if not chosen:
                empty = {"node": node_id, "problem_count": 0, "first_line": 0, "first_reason": "no_samples"}
                raise LocalDatasetInvalidError(f"no {name} samples after the split", params=empty)
            folder = self.host_path(ref.path)
            for record in chosen:  # a link inside the folder may not point out (security R1)
                try:
                    image_path(folder, record)
                except ValueError:
                    # first_line 0: after filter / dedupe / split the record order is not labels.jsonl's line order
                    bad = {"node": node_id, "problem_count": 1, "first_line": 0, "first_reason": "bad_image_name"}
                    raise LocalDatasetInvalidError(f"{name} image outside its folder", params=bad) from None
            prompts[name] = single_prompt(chosen, family)
            return self.container_path(write_meta_dataset(out, name, chosen, family, media_dir=ref.path))

        train_meta = meta(train, "train")
        val_meta = meta(val, "validation") if val is not None and val.records else None
        return train_meta, val_meta, prompts["train"]

    def _merge_lora(self, job: NodeJob) -> NodeResult:
        """LoRA adapter + its base -> one HF checkpoint (CPU job in the training image; B-1)."""
        base, adapter = job.inputs["base_model"], job.inputs["adapter"]
        entry = self._entry_for(base)
        run_dir = job.run_dir.resolve()
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        out = run_dir / "merged"
        run_id = run_dir.name[:120]
        manifest = RunManifest(
            run_id=run_id, name=f"{job.spec.type_id} {entry.display_name}", stage="train", task="merge_lora",
            base_model=entry.repo, base_model_revision=entry.revision, container_image=self.image,
            parent_run_id=_run_id_of(adapter),
        )  # fmt: skip
        container = ContainerJob(
            name=run_id,
            image=self.image,
            argv=(
                "python",
                "-m",
                "vlm_pipeline.train.merge_lora",
                "--base",
                self.container_path(self.model_dir(base)),
                "--adapter",
                self.container_path(self.host_path(adapter.path)),
                "--out",
                self.container_path(out),
                "--auto-class",
                entry.auto_class,
                "--dtype",
                _merge_dtype(entry),
            ),
            gpu=False,
            read_only_project=True,  # the merge writes only its own run dir
            rw_dirs=(run_dir,),
            narrow=True,
            ro_inputs=(self.mount_input(self.model_dir(base)), self.mount_input(adapter.path)),
        )
        try:
            final = run_job(
                container, manifest, run_dir, project_root=self.root, who=self.who, est_peak_gb=_merge_peak_gb(entry),
                ckpt_gb=entry.disk_gb, cancel=job.cancel, on_wait=job.on_wait,
            )  # fmt: skip
        except MemoryPreemptedBeforeStartError as exc:
            raise InsufficientMemoryError(str(exc), params={"need_gb": _merge_peak_gb(entry)}) from exc
        if final.status != RunStatus.COMPLETED:
            raise ContainerExit(final.exit_code or 1, oom_killed=bool(final.oom_killed),
                                cancelled_by_executor=final.status == RunStatus.CANCELLED,
                                log_tail=log_tail(run_dir))  # fmt: skip
        from vlm_pipeline.train.ckpt_complete import complete_from_base

        # Keys the HF class does not build (e.g. Qwen3.5 MTP head) are dropped on save: restore them from the base.
        complete_from_base(self.model_dir(base), out)
        return NodeResult(
            outputs={"model": ArtifactRef(type="Checkpoint", path=str(out), hash=f"{run_id}:{adapter.hash}")}
        )

    @staticmethod
    def _write_metrics(run_dir: Path) -> None:
        for src, split in (("training.jsonl", "train"), ("validation.jsonl", "val")):
            path = run_dir / "checkpoints" / src
            if path.exists():
                lines = automodel_to_metrics(path.read_text(encoding="utf-8").splitlines(), split=split)  # type: ignore[arg-type]
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                    fh.writelines(line + "\n" for line in lines)


def _trainable_env(entry: CatalogEntry, params: LoraParams | FullFinetuneParams) -> dict[str, str]:
    """Patch-0002 guard: the job fails before step 1 if the trainable share is not what the catalog expects."""
    if not isinstance(params, FullFinetuneParams) or params.train_blocks != "lm":  # the catalog % is for lm only
        return {}
    expected = entry.sft_trainable_pct
    if expected is None:
        logger.warning("no sft_trainable_pct in the catalog for %s: freeze guard disabled", entry.id)
        return {}
    return {"VLM_EXPECTED_TRAINABLE_PCT": f"{expected[0]}:{expected[1]}"}


def _set_aside_checkpoints(run_dir: Path, model_id: str) -> None:
    """Start a model that cannot resume over: move its checkpoints and metrics aside (kept, never deleted).

    The run dir is stable per cache key and AutoModel resumes from any checkpoint in it, so a failing resume would
    repeat on every retry.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = False
    for name in ("checkpoints", "metrics.jsonl"):
        if (run_dir / name).exists():
            (run_dir / name).rename(run_dir / f"{name}.not-resumable-{stamp}")
            moved = True
    if moved:
        logger.warning("%s cannot resume from a checkpoint yet: training starts over in %s", model_id, run_dir)


MERGE_FP32_MAX_PARAMS_B = 4.0  # fp32 merge (exact delta) while the model fits the shared host comfortably


def _check_train(q: SupportQuery) -> None:
    if q.model is not None and not q.model.supports_train_node(q.spec.type_id):
        raise TrainModeUnsupportedError(
            "train node not vetted for this model",
            params={"node": q.node_id, "model": q.model.display_name, "supported": q.model.train_node},
        )
    advanced = getattr(q.params, "advanced", {})
    if q.model is not None and (wrong := keys_not_for(q.model.id, advanced)):  # P016 §3: optimizers differ per model
        raise AdvancedNotForModelError(
            "advanced option not in this model's template",
            params={"node": q.node_id, "param": "advanced", "key": wrong[0], "rule": "not_for_model"},
        )
    blocks = getattr(q.params, "train_blocks", "lm")
    if blocks != "lm" and q.model is not None and not _has_block_prefixes(q.model, blocks):
        raise TrainBlocksNotForModelError(  # P023 §1: the recipe could not say which modules to freeze / train
            "train_blocks needs module prefixes this model does not have",
            params={"node": q.node_id, "param": "train_blocks", "value": blocks, "rule": "not_for_model", "key": ""},
        )
    quantize = getattr(q.params, "quantize_base", "none")
    if quantize != "none" and q.model is not None and q.model.id not in qlora_models():
        raise QloraNotForModelError(  # QLoRA only where the template names the modules that stay bf16
            "QLoRA is not set up for this model",
            params={"node": q.node_id, "param": "quantize_base", "value": quantize, "rule": "not_for_model", "key": ""},
        )


def _has_block_prefixes(entry: CatalogEntry, train_blocks: str) -> bool:
    """Whether build_recipe can map ``train_blocks`` for this model (the same lists it receives)."""
    prefixes = block_prefixes(entry)
    if prefixes is None:
        return False
    vision, projector = prefixes
    return bool(vision) if train_blocks == "all" else bool(projector)


def _model_blocks(entry: CatalogEntry) -> Blocks:
    prefixes = block_prefixes(entry)
    return Blocks(*prefixes) if prefixes is not None else Blocks()


def _check_merge(q: SupportQuery) -> None:
    if q.model is not None and q.model.auto_class != "AutoModelForImageTextToText":  # remote code never trains LoRA
        raise NodeNotRunnableError(
            "merge needs a native model", params={"node": q.node_id, "node_type": q.spec.type_id}
        )


GRPO_BLOCKED_REASON = "engine_single_gpu"  # NeMo-RL v0.7.0 has no 1-GPU VLM GRPO recipe (research probe, 2026-10-14)


def _check_grpo(q: SupportQuery) -> None:
    """GRPO ships as "coming soon" in 1.0 (PM 2026-10-14): refused on the canvas and at run, with the real reason.

    Lifted only when the NeMo-RL probe passes (a real 1-GPU GRPO run on GB10, reward going up, RAM within budget).
    """
    raise NodeNotRunnableError(
        "GRPO has no single-GPU engine configuration for this machine yet",
        params={"node": q.node_id, "node_type": q.spec.type_id, "reason": GRPO_BLOCKED_REASON},
    )


DPO_BLOCKED_REASON = "engine_no_vlm_dpo"  # NeMo-RL v0.7.0 DPO is text-only; research patches the VLM path (ADR-022 #7)


def _check_dpo(q: SupportQuery) -> None:
    """Refused on the canvas and at run with its real reason until the multimodal DPO patch passes (research)."""
    raise NodeNotRunnableError(
        "DPO has no vision-language engine path yet",
        params={"node": q.node_id, "node_type": q.spec.type_id, "reason": DPO_BLOCKED_REASON},
    )


def _check_export(q: SupportQuery) -> None:
    from vlm_pipeline.app.export_worker import ExportLicenseUnknownError, ExportUnsupportedError

    if q.model is not None and q.model.remote_code != "none":  # the folder's real content is re-checked at run time
        raise ExportUnsupportedError(
            "model runs its own code", params={"node": q.node_id, "model": q.model.display_name}
        )
    if q.model is not None and q.model.tier == "user" and normalize_license(q.model.license) == LICENSE_UNKNOWN:
        raise ExportLicenseUnknownError(
            "no licence named for this model", params={"node": q.node_id, "name": q.model.display_name,
                                                       "repo": q.model.repo})  # fmt: skip


def _owner_check(module: str) -> Callable[[SupportQuery], None]:
    """The node owner's pure ``check_support(node_id, params, model, inputs)``.

    Plain arguments, not SupportQuery: eval and quantize may only import common (ADR-004 A3); app modules use the
    same signature for one convention. A missing function fails at import, never silently disables a check.
    """
    import importlib

    fn = getattr(importlib.import_module(module), "check_support", None)
    if not callable(fn):
        raise ImportError(f"{module}.check_support is missing: the node's support check would be silently off")
    return lambda q: fn(q.node_id, q.params, q.model, q.inputs)


def _support_checks() -> dict[str, Callable[[SupportQuery], None]]:
    return {
        "train.lora_vlm": _check_train,
        "train.sft_vlm": _check_train,
        "train.merge_lora": _check_merge,
        "train.grpo_vlm": _check_grpo,  # "coming soon": no handler, refused with its reason (P020)
        "train.dpo_vlm": _check_dpo,  # ADR-022 #7: refused with its reason until the VLM DPO patch
        "quantize.qat": _owner_check("vlm_pipeline.quantize.qat"),  # ADR-022 #10 (fixcode)
        "data.caption_vlm": _owner_check("vlm_pipeline.app.caption_worker"),
        "data.filter": _owner_check("vlm_pipeline.app.data_workers"),
        "data.dedupe": _owner_check("vlm_pipeline.app.data_workers"),
        "export.hf_dir": _check_export,
        "quantize.modelopt_ptq": _owner_check("vlm_pipeline.quantize.scope"),
        "eval.benchmarks": _owner_check("vlm_pipeline.eval.node_worker"),
        "bench.perf": _owner_check("vlm_pipeline.app.bench_worker"),
        "serve.vllm": _owner_check("vlm_pipeline.app.trial_worker"),
    }


def _merge_dtype(entry: CatalogEntry) -> str:
    return "fp32" if entry.params_b <= MERGE_FP32_MAX_PARAMS_B else "bf16"


def _merge_peak_gb(entry: CatalogEntry) -> float:
    """One copy of the weights in the merge dtype (+30 %) plus 4 GB runtime, at least 10 GB.

    The merge loads the model once and saves it shard by shard (never two full copies). Measured on 26.08-6
    (research 2026-10-13): Qwen3.5-2B fp32 9.5 GB, 4B bf16 8.5 GB; the old "x 2" asked 9B for 55 GB and blocked it.
    peak_mem_gb under-reads mmap'd weights (page cache counts as available): never go below one copy of the weights.
    """
    per_param = 4 if _merge_dtype(entry) == "fp32" else 2
    return round(max(10.0, entry.params_b * per_param * 1.3 + 4), 1)


def _run_id_of(artifact: ArtifactRef) -> str | None:
    """Run id of a trained artifact (hash ``<run_id>:<config_hash>``), for run.json lineage."""
    head = artifact.hash.split(":", 1)[0]
    return head if ":" in artifact.hash and "@" not in head else None


__all__ = [
    "DEFAULT_TRAIN_IMAGE",
    "PREPARED_DIRS",
    "NodeNotRunnableError",
    "NotCachedError",
    "RealWorker",
    "TrainModeUnsupportedError",
    "UnknownModelError",
]

_SUPPORT_CHECKS = _support_checks()
