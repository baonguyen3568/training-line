"""Real node worker for the app (``--worker container``): host source nodes + AutoModel fine-tune jobs.

Implements ``graph.worker.Worker``. Heavy nodes go through ``common.jobrun.run_job`` so every job gets the preflight,
the GPU lock, the host memory watchdog and run.json. Errors are raised, never classified here (the executor maps
them to catalog codes). Nothing is downloaded: models/datasets must already be in the project HF cache (offline jobs).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

from vlm_pipeline.common.catalog import Catalog, CatalogEntry, load_catalog
from vlm_pipeline.common.errors import InsufficientMemoryError, VlmPipelineError
from vlm_pipeline.common.jobrun import (
    CONTAINER_ROOT,
    ContainerJob,
    MemoryPreemptedBeforeStartError,
    run_job,
)
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, automodel_to_metrics
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob, NodeResult
from vlm_pipeline.train.config import FullFinetuneParams, LoraParams
from vlm_pipeline.train.recipe import build_recipe, load_template, render_recipe, template_path

DEFAULT_TRAIN_IMAGE = "vlm-automodel:26.08-2"
# Hub datasets vetted by bao-security, pinned revisions. TODO(research): move into a datasets catalog file.
HUB_DATASET_REVISIONS = {"naver-clova-ix/cord-v2": "7f0115a4b758a71d6473b8d085751692da2fef98"}
# Expected trainable-parameter share for full fine-tune with the default freeze (fixcode patch 0002 guard).
# TODO(research): move into the model catalog next to freeze_* fields.
EXPECTED_TRAINABLE_PCT_SFT = {"nemotron-parse-v1.1": "29.4:31.4", "nemotron-parse-2.0": "26.8:28.8"}
PREPARED_DIRS = {"nemotron-parse-v1.1": "nemotron-parse-v1.1_fd1e3c6b"}  # train.prepare outputs (hash-verified)


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

    # ------------------------------------------------------------------------------------------ helpers
    def _container_path(self, host_path: Path) -> str:
        return f"{CONTAINER_ROOT}/{host_path.resolve().relative_to(self.root).as_posix()}"

    def _host_path(self, container_path: str) -> Path:
        return self.root / container_path.removeprefix(CONTAINER_ROOT + "/")

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
        ref = ArtifactRef(type="Model", path=self._container_path(path), hash=f"{entry.repo}@{entry.revision}")
        return NodeResult(outputs={"model": ref})

    def _hub_dataset(self, job: NodeJob) -> NodeResult:
        repo = str(getattr(job.params, "repo"))  # noqa: B009
        revision = HUB_DATASET_REVISIONS.get(repo)
        if revision is None:
            raise NotCachedError("dataset has no pinned revision", params={"repo": repo, "reason": "not_pinned"})
        path = _hub_snapshot(self.root, repo, revision, "dataset")
        ref = ArtifactRef(type="Dataset", path=self._container_path(path), hash=f"{repo}@{revision}")
        return NodeResult(outputs={"dataset": ref})

    def _train(self, job: NodeJob) -> NodeResult:
        params = job.params
        if not isinstance(params, LoraParams | FullFinetuneParams):
            raise NodeNotRunnableError("unexpected params model", params={"node_type": job.spec.type_id})
        base, data = job.inputs["base_model"], job.inputs["train_data"]
        entry = self._entry_for(base)
        class_token = None
        if entry.loss == "nemotron_parse":
            from vlm_pipeline.train.recipe_checks import resolve_parse_class_token_start_idx

            class_token = resolve_parse_class_token_start_idx(self._host_path(base.path))
        run_dir = job.run_dir.resolve()
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        recipe = build_recipe(
            load_template(template_path(self.root, entry)),
            params,
            model_dir=base.path,
            dataset_dir=data.path,
            run_dir=self._container_path(run_dir),
            class_token_start_idx=class_token,
        )
        text = render_recipe(recipe)
        (run_dir / "config.yaml").write_text(text, encoding="utf-8")
        config_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        run_id = f"{run_dir.parent.name}-{job.node_id}"[:120]
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
        ref = ArtifactRef(type=out_type.type, path=self._container_path(out), hash=f"{run_id}:{config_hash}")
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
    expected = EXPECTED_TRAINABLE_PCT_SFT.get(entry.id)
    if expected is None or not isinstance(params, FullFinetuneParams) or not params.freeze_vision_tower:
        return {}
    return {"VLM_EXPECTED_TRAINABLE_PCT": expected}


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


__all__ = ["DEFAULT_TRAIN_IMAGE", "NodeNotRunnableError", "NotCachedError", "RealWorker"]
