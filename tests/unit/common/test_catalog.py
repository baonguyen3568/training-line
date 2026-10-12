"""Model/dataset catalog loaders: the real files load, and malformed entries are rejected."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vlm_pipeline.common.catalog import CatalogEntry, DatasetEntry, load_catalog, load_dataset_catalog


def test_real_catalogs_load() -> None:
    models = load_catalog()
    assert models.default.tier == "default"
    parse = models.get("nemotron-parse-v1.1")
    assert parse is not None
    assert parse.sft_trainable_pct == (29.4, 31.4)
    cord = load_dataset_catalog().get("cord-v2")
    assert cord is not None
    assert "test" not in cord.trainable_splits


def _model(**override: Any) -> dict[str, Any]:
    entry = load_catalog().default.model_dump()
    entry.update(override)
    return entry


@pytest.mark.parametrize("repo", ["../etc", "org/..", ".org/x", "org/.x", "org", "a/b/c"])
def test_repo_id_cannot_traverse(repo: str) -> None:
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate(_model(repo=repo))


@pytest.mark.parametrize(
    ("low", "high", "ok"), [(0, 0, True), (29.4, 31.4, True), (31.4, 29.4, False), (0, 30, False), (10, 101, False)]
)
def test_trainable_pct_range(low: float, high: float, ok: bool) -> None:
    raw = _model(sft_trainable_pct_min=low, sft_trainable_pct_max=high)
    if ok:
        CatalogEntry.model_validate(raw)
    else:
        with pytest.raises(ValidationError):
            CatalogEntry.model_validate(raw)


def test_unknown_field_and_unpinned_revision_rejected() -> None:
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate(_model(freeze_embeddings=True))
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate(_model(revision="main"))


def test_eval_only_dataset_is_never_trainable_and_heldout_must_exist() -> None:
    cord = load_dataset_catalog().datasets[0].model_dump()
    assert DatasetEntry.model_validate({**cord, "use": "eval_only"}).trainable_splits == ()
    with pytest.raises(ValidationError):
        DatasetEntry.model_validate({**cord, "heldout_splits": ["nope"]})
