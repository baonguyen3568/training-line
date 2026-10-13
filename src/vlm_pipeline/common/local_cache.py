"""Is a catalog model / dataset on this machine? One answer for the run (raises NET-002) and validate (warns NET-002).

The run never downloads (jobs have no network, ADR-015); what is missing comes from the app's download (G-5, P018).
The ``params`` returned are the NET-002 params the UI uses: ``catalog_id`` + ``kind`` feed ``POST /api/downloads``,
``name`` + ``license`` + ``size_gb`` the "download before running" dialog (its license checkbox lists them all).
"""

from __future__ import annotations

from pathlib import Path

from vlm_pipeline.common.catalog import CatalogEntry, DatasetEntry

PREPARED_DIRS = {"nemotron-parse-v1.1": "nemotron-parse-v1.1_fd1e3c6b"}  # train.prepare outputs (hash-verified)


def hub_snapshot(root: Path, kind: str, repo: str, revision: str) -> Path:
    prefix = "models" if kind == "model" else "datasets"
    return root / ".cache" / "hf" / "hub" / f"{prefix}--{repo.replace('/', '--')}" / "snapshots" / revision


def prepared_model_dir(root: Path, catalog_id: str) -> Path | None:
    """The pinned prepared copy of a remote-code model (``pinned_prepared``), or None if the catalog has none."""
    name = PREPARED_DIRS.get(catalog_id)
    return root / ".cache" / "prepared" / name / "model" if name else None


def eval_data_dir(root: Path, entry: DatasetEntry) -> Path:
    """``<root>/data/eval/<id>-<rev8>``: where eval data extracted from a catalog dataset lives (eval nodes read it)."""
    return root / "data" / "eval" / f"{entry.id}-{entry.revision[:8]}"


def model_missing(root: Path, entry: CatalogEntry) -> dict[str, object] | None:
    """NET-002 params when ``entry`` cannot run from this machine's cache, else None."""
    params: dict[str, object] = {"repo": entry.repo, "catalog_id": entry.id, "kind": "model", "size_gb": entry.disk_gb,
                                 "name": entry.display_name, "license": entry.license}  # fmt: skip
    if entry.remote_code == "pinned_prepared":
        prepared = prepared_model_dir(root, entry.id)
        if prepared is None or not (prepared / "config.json").is_file():
            return {**params, "reason": "not_prepared"}
        return None
    if not hub_snapshot(root, "model", entry.repo, entry.revision).is_dir():
        return {**params, "reason": "not_cached"}
    return None


def dataset_missing(root: Path, entry: DatasetEntry) -> dict[str, object] | None:
    if hub_snapshot(root, "dataset", entry.repo, entry.revision).is_dir():
        return None
    return {"repo": entry.repo, "reason": "not_cached", "catalog_id": entry.id, "kind": "dataset",
            "size_gb": entry.disk_gb, "name": entry.display_name, "license": entry.license}  # fmt: skip


__all__ = ["PREPARED_DIRS", "dataset_missing", "eval_data_dir", "hub_snapshot", "model_missing", "prepared_model_dir"]
