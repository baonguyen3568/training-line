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
import time
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.catalog import Catalog, CatalogEntry, load_catalog, load_dataset_catalog
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
# Models whose AutoModel checkpoints cannot be resumed yet. Parse: RADIO drops its input_conditioner buffers at the
# first forward, so the saved model shard lacks keys that a freshly built model expects ("Missing key in checkpoint
# state_dict", resume smoke 2026-10-13). A resume of these starts over instead of failing on every retry.
RESUME_UNSUPPORTED = frozenset({"nemotron-parse-v1.1"})


class NotCachedError(VlmPipelineError):
    """A model or dataset is not in the project cache; jobs run offline, so it must be downloaded first."""

    code: ClassVar[str] = "NET-001"


class NodeNotRunnableError(VlmPipelineError):
    """This worker has no implementation for the node type yet."""

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
        raise NodeNotRunnableError("base model is not a catalog model", params={"hash": ref.hash})

    # ------------------------------------------------------------------------------------------ nodes
    def run(self, job: NodeJob) -> NodeResult:
        handler = {
            "model.hf_pick": self._model_pick,
            "data.hf_load": self._hub_dataset,
            "train.lora_vlm": self._train,
            "train.sft_vlm": self._train,
        }.get(job.spec.type_id)
        if handler is None:
            raise NodeNotRunnableError(f"no worker for {job.spec.type_id}", params={"node_type": job.spec.type_id})
        return handler(job)

    def _model_pick(self, job: NodeJob) -> NodeResult:
        model_id = str(getattr(job.params, "model"))  # noqa: B009 - dynamic params model (catalog Literal)
        entry = self.catalog.get(model_id)
        if entry is None:
            raise NodeNotRunnableError("unknown catalog model", params={"model": model_id})
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
        # The artifact is a small descriptor (snapshot + sample limit), so "quick run" sizes cost no copy.
        job.run_dir.mkdir(parents=True, exist_ok=True)
        descriptor = job.run_dir / "dataset.json"
        descriptor.write_text(
            json.dumps({"repo": repo, "revision": revision, "path": self._container_path(path), "limit": limit}),
            encoding="utf-8",
        )
        digest = f"{repo}@{revision}" + (f"#limit={limit}" if limit else "")
        ref = ArtifactRef(type="Dataset", path=str(descriptor.resolve()), hash=digest)
        return NodeResult(outputs={"dataset": ref})

    def _dataset_input(self, ref: ArtifactRef) -> tuple[str, int]:
        """(container dataset dir, sample limit) from a dataset artifact (descriptor json or plain dir)."""
        if ref.path.endswith(".json"):
            desc = json.loads(self._host_path(ref.path).read_text(encoding="utf-8"))
            return str(desc["path"]), int(desc.get("limit", 0))
        return self._container_path(self._host_path(ref.path)), 0

    def _train(self, job: NodeJob) -> NodeResult:
        params = job.params
        if not isinstance(params, LoraParams | FullFinetuneParams):
            raise NodeNotRunnableError("unexpected params model", params={"node_type": job.spec.type_id})
        base, data = job.inputs["base_model"], job.inputs["train_data"]
        entry = self._entry_for(base)
        # Full-fine-tune-only models (Parse: FusedAdam fp32-master recipe, no LoRA target modules) were never run with
        # LoRA: fail before the GPU lock. TODO(bao-fixcode): own code CFG-010 + the same rule in graph validate.
        if entry.train_node == "train.sft_vlm" and job.spec.type_id == "train.lora_vlm":
            raise NodeNotRunnableError(
                "train mode not supported for this model",
                params={"node_type": job.spec.type_id, "model": entry.id, "supported": entry.train_node},
            )
        class_token = None
        if entry.loss == "nemotron_parse":
            from vlm_pipeline.train.recipe_checks import resolve_parse_class_token_start_idx

            class_token = resolve_parse_class_token_start_idx(self._host_path(base.path))
        run_dir = job.run_dir.resolve()
        if entry.id in RESUME_UNSUPPORTED:
            _set_aside_checkpoints(run_dir, entry.id)
            (run_dir / NOT_RESUMABLE_MARKER).touch()  # executor: no "resume" offer, warn on the next run
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        dataset_dir, limit = self._dataset_input(data)
        recipe = build_recipe(
            load_template(template_path(self.root, entry)),
            params,
            model_dir=self._container_path(self._host_path(base.path)),
            dataset_dir=dataset_dir,
            train_limit=limit or None,
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


__all__ = ["DEFAULT_TRAIN_IMAGE", "NodeNotRunnableError", "NotCachedError", "RealWorker"]
