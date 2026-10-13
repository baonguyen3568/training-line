"""Real node worker for the app (``--worker container``): host source nodes + AutoModel fine-tune jobs.

Implements ``graph.worker.Worker``. Heavy nodes go through ``common.jobrun.run_job`` so every job gets the preflight,
the GPU lock, the host memory watchdog and run.json. Errors are raised, never classified here (the executor maps
them to catalog codes). Nothing is downloaded: models/datasets must already be in the project HF cache (offline jobs).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.app.trial_worker import run_trial
from vlm_pipeline.common.catalog import Catalog, CatalogEntry, load_catalog, load_dataset_catalog
from vlm_pipeline.common.dataset_ref import DatasetRef, read_dataset_ref, write_dataset_ref
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.images import DEFAULT_TRAIN_IMAGE
from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    MemoryPreemptedBeforeStartError,
    run_job,
)
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, automodel_to_metrics
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import NOT_RESUMABLE_MARKER, ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.nodes.spec import NodeSpec
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams
from vlm_pipeline.train.recipe import build_recipe, load_template, render_recipe, template_path

logger = logging.getLogger(__name__)
PREPARED_DIRS = {"nemotron-parse-v1.1": "nemotron-parse-v1.1_fd1e3c6b"}  # train.prepare outputs (hash-verified)
# Models whose AutoModel checkpoints cannot be resumed: a resume starts over instead of failing on every retry.
# Empty since vlm-automodel:26.08-6 (patch 0005, BUG-003 closed: Parse resumes, also from 26.08-5 checkpoints).
# Run dirs marked .not-resumable earlier stay so (graph.worker.NOT_RESUMABLE_MARKER).
RESUME_UNSUPPORTED: frozenset[str] = frozenset()


class NotCachedError(VlmPipelineError):
    """A model or dataset is not in the project cache; jobs run offline, so it must be downloaded first."""

    code: ClassVar[str] = "NET-001"


class TrainModeUnsupportedError(VlmPipelineError):
    """The fine-tune node is not the one the catalog vetted for this model (raised before the GPU lock)."""

    code: ClassVar[str] = "CFG-010"


class NodeNotRunnableError(VlmPipelineError):
    """This worker has no implementation for the node type yet (validate reports it first via ``supports``)."""

    code: ClassVar[str] = "CFG-011"


class UnknownModelError(VlmPipelineError):
    """A model reference that is not a catalog model (validated params make this an internal error)."""

    code: ClassVar[str] = "SYS-000"


def _hub_snapshot(root: Path, repo: str, revision: str, kind: str) -> Path:
    prefix = "models" if kind == "model" else "datasets"
    path = root / ".cache" / "hf" / "hub" / f"{prefix}--{repo.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise NotCachedError(f"{repo}@{revision} not in cache", params={"repo": repo, "reason": "not_cached"})
    return path


class RealWorker:
    def __init__(
        self,
        root: Path,
        *,
        catalog: Catalog | None = None,
        image: str = DEFAULT_TRAIN_IMAGE,
        who: str = "vlm-app",
    ) -> None:
        self.root = root.resolve()
        self.catalog = catalog or load_catalog()
        self.image = image
        self.who = who

    def identity(self, spec: NodeSpec) -> str:
        """Part of the executor cache key: container nodes are keyed by the exact image (content id)."""
        if spec.runner != "container":
            return "host/real-1"
        return f"container/{self.image}@{self._image_id()}"

    @functools.cache  # noqa: B019 - one worker per app process; the image id does not change while it runs
    def _image_id(self) -> str:
        from vlm_pipeline.common.jobrun import image_digest

        return image_digest(self.image) or "unknown"

    # ------------------------------------------------------------------------------------------ helpers
    def _container_path(self, host_path: Path) -> str:
        return f"{CONTAINER_ROOT}/{host_path.resolve().relative_to(self.root).as_posix()}"

    def _host_path(self, path: str) -> Path:
        """Artifact paths are host paths; container paths (older entries) are mapped back."""
        if path.startswith(CONTAINER_ROOT + "/"):
            return self.root / path.removeprefix(CONTAINER_ROOT + "/")
        return Path(path)

    def _entry_for(self, ref: ArtifactRef) -> CatalogEntry:
        for entry in self.catalog.models:
            if ref.hash == f"{entry.repo}@{entry.revision}":
                return entry
        raise UnknownModelError("base model is not a catalog model", params={"hash": ref.hash})

    # ------------------------------------------------------------------------------------------ nodes
    @functools.cached_property
    def _handlers(self) -> dict[str, Callable[[NodeJob], NodeResult]]:
        """Node types this worker really runs; every other type is refused at validate time (CFG-011)."""
        return {
            "model.hf_pick": self._model_pick,
            "data.hf_load": self._hub_dataset,
            "data.split": self._split,
            "train.lora_vlm": self._train,
            "train.sft_vlm": self._train,
            "serve.vllm": functools.partial(run_trial, self),  # B-7 "Chạy thử" (app/trial_worker.py)
        }

    def supports(self, spec: NodeSpec) -> bool:
        return spec.type_id in self._handlers

    def run(self, job: NodeJob) -> NodeResult:
        handler = self._handlers.get(job.spec.type_id)
        if handler is None:
            raise NodeNotRunnableError(
                f"no worker for {job.spec.type_id}", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        return handler(job)

    def _model_pick(self, job: NodeJob) -> NodeResult:
        model_id = str(getattr(job.params, "model"))  # noqa: B009 - dynamic params model (catalog Literal)
        entry = self.catalog.get(model_id)
        if entry is None:
            raise UnknownModelError("unknown catalog model", params={"model": model_id})
        if entry.remote_code == "pinned_prepared":
            prepared = PREPARED_DIRS.get(entry.id)
            path = self.root / ".cache" / "prepared" / (prepared or "") / "model"
            if prepared is None or not (path / "config.json").is_file():
                raise NotCachedError("prepared snapshot missing", params={"repo": entry.repo, "reason": "not_prepared"})
        else:
            path = _hub_snapshot(self.root, entry.repo, entry.revision, "model")
        ref = ArtifactRef(type="Model", path=str(path.resolve()), hash=f"{entry.repo}@{entry.revision}")
        return NodeResult(outputs={"model": ref})

    def _hub_dataset(self, job: NodeJob) -> NodeResult:
        repo = str(getattr(job.params, "repo"))  # noqa: B009
        entry = next((d for d in load_dataset_catalog().datasets if d.repo == repo), None)
        if entry is None:
            raise NotCachedError("dataset not in the catalog", params={"repo": repo, "reason": "not_pinned"})
        revision = entry.revision
        path = _hub_snapshot(self.root, repo, revision, "dataset")
        limit = int(getattr(job.params, "max_samples", 0) or 0)
        # The artifact is a small descriptor (snapshot + split + sample limit), so "quick run" sizes cost no copy.
        ref = DatasetRef(repo=repo, revision=revision, path=self._container_path(path), limit=limit)
        return NodeResult(outputs={"dataset": self._dataset_artifact(job.run_dir / "dataset.json", ref)})

    def _dataset_artifact(self, path: Path, ref: DatasetRef) -> ArtifactRef:
        write_dataset_ref(path, ref)
        return ArtifactRef(type="Dataset", path=str(path.resolve()), hash=ref.hash)

    def _dataset_input(self, ref: ArtifactRef) -> DatasetRef:
        """The descriptor behind a Dataset artifact (a plain folder reads as all of it, train split)."""
        if ref.path.endswith(".json"):
            return read_dataset_ref(self._host_path(ref.path))
        return DatasetRef(path=self._container_path(self._host_path(ref.path)))

    def _split(self, job: NodeJob) -> NodeResult:
        """Train / val of a dataset; held-out test splits never appear.

        Hub datasets with an official validation split use it (results stay comparable with published numbers);
        ``val_fraction`` applies to data without one.
        """
        src = self._dataset_input(job.inputs["dataset"])
        entry = next((d for d in load_dataset_catalog().datasets if src.repo and d.repo == src.repo), None)
        if entry is None or "validation" not in entry.splits or "validation" in entry.heldout_splits:
            raise NodeNotRunnableError(  # local folders: fraction split comes with data.local_folder (B-3)
                "split by fraction not implemented yet", params={"node": job.node_id, "node_type": job.spec.type_id}
            )
        train = src.model_copy(update={"split": "train"})
        val = src.model_copy(update={"split": "validation", "limit": 0})  # quick-run size limits training only
        return NodeResult(
            outputs={
                "train": self._dataset_artifact(job.run_dir / "train.json", train),
                "val": self._dataset_artifact(job.run_dir / "val.json", val),
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
        if not entry.supports_train_node(job.spec.type_id):  # graph validate blocks it first; this is the backstop
            raise TrainModeUnsupportedError(
                "train node not vetted for this model",
                params={"node": job.node_id, "model": entry.display_name, "supported": entry.train_node},
            )
        class_token = None
        if entry.loss == "nemotron_parse":
            from vlm_pipeline.train.recipe_checks import resolve_parse_class_token_start_idx

            class_token = resolve_parse_class_token_start_idx(self._host_path(base.path))
        run_dir = job.run_dir.resolve()
        if entry.id in RESUME_UNSUPPORTED:
            _set_aside_checkpoints(run_dir, entry.id)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        if entry.id in RESUME_UNSUPPORTED:
            (run_dir / NOT_RESUMABLE_MARKER).touch()  # executor: no "resume" offer, warn on the next run
        train = self._dataset_input(data)
        val = self._dataset_input(job.inputs["val_data"]) if "val_data" in job.inputs else None
        template = load_template(template_path(self.root, entry))
        if val is not None and "validation_dataset" not in template:
            logger.warning("%s: template has no validation_dataset; the wired val_data is not used", entry.id)
        source = next((d for d in load_dataset_catalog().datasets if train.repo and d.repo == train.repo), None)
        no_val_split = source is None or "validation" not in source.splits or "validation" in source.heldout_splits
        recipe = build_recipe(
            template,
            params,
            model_dir=self._container_path(self._host_path(base.path)),
            dataset_dir=train.path,
            train_limit=train.limit or None,
            train_split=train.split,
            val_dataset_dir=val.path if val else None,
            val_split=val.split if val else None,
            drop_validation=val is None and no_val_split,
            run_dir=self._container_path(run_dir),
            class_token_start_idx=class_token,
        )
        text = render_recipe(recipe)
        (run_dir / "config.yaml").write_text(text, encoding="utf-8")
        config_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        run_id = run_dir.name[:120]  # executor run dir = <node>-<cache key prefix>: unique, stable on resume
        manifest = RunManifest(
            run_id=run_id,
            name=f"{job.spec.type_id} {entry.display_name}",
            stage="train",
            task="lora" if isinstance(params, LoraParams) else "sft",
            base_model=entry.repo,
            base_model_revision=entry.revision,
            dataset=data.hash,
            config_path=self._container_path(run_dir / "config.yaml"),
            config_hash=config_hash,
            seed=params.seed,
            container_image=self.image,
        )
        container = ContainerJob(
            name=run_id,
            image=self.image,
            argv=("automodel", self._container_path(run_dir / "config.yaml"), "--nproc-per-node", "1"),
            workdir="/opt/Automodel",
            env=_trainable_env(entry, params),
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
                log_tail=_tail(run_dir / "logs" / "stdout.log"),
            )
        port, out_type = next(iter(job.spec.outputs.items()))
        out = run_dir / "checkpoints" / "LATEST" / "model"
        if (out / "consolidated").is_dir():  # full fine-tune: HF safetensors loadable by downstream nodes
            out = out / "consolidated"
            from vlm_pipeline.train.ckpt_complete import complete_from_base

            # BUG-002: consolidation drops buffers (RADIO input conditioner) -> restore them from the base model.
            complete_from_base(self._host_path(base.path), out)
        ref = ArtifactRef(type=out_type.type, path=str(out), hash=f"{run_id}:{config_hash}")
        return NodeResult(outputs={port: ref})

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
    if not isinstance(params, FullFinetuneParams) or not params.freeze_vision_tower:
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


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


__all__ = [
    "DEFAULT_TRAIN_IMAGE",
    "NodeNotRunnableError",
    "NotCachedError",
    "RealWorker",
    "TrainModeUnsupportedError",
    "UnknownModelError",
]
