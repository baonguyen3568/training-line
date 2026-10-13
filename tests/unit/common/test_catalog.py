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


def test_every_catalog_licence_has_an_approved_locked_text() -> None:
    from vlm_pipeline.common.licenses import license_text
    from vlm_pipeline.common.paths import project_root

    models = load_catalog().models
    ids = {m.license for m in models} | {p.license for m in models for p in m.model_parts}
    ids |= {d.license for d in load_dataset_catalog().datasets}
    missing = sorted(i for i in ids if license_text(project_root(), i) is None)
    assert missing == [], f"add configs/licenses/<id>.txt + an APPROVED row in LICENSES.lock for {missing}"


def test_notices_and_model_parts_are_strict() -> None:
    from vlm_pipeline.common.catalog import ModelPart

    part = ModelPart(name="tokenizer", license="cc-by-4.0", notices=("Tokenizer from {repo}@{revision}.",))
    assert part.notices == ("Tokenizer from {repo}@{revision}.",)
    with pytest.raises(ValidationError):
        ModelPart.model_validate({"name": "tokenizer", "license": "cc-by-4.0", "notice": "typo"})
    with pytest.raises(ValidationError):
        ModelPart(name="", license="cc-by-4.0")


def test_licence_lock_is_fail_closed(tmp_path: Any) -> None:
    import hashlib

    from vlm_pipeline.common.licenses import LICENSES_DIR, LICENSES_LOCK, license_text

    (tmp_path / LICENSES_DIR).mkdir(parents=True)
    for lic in ("ok", "pending", "tampered"):
        (tmp_path / LICENSES_DIR / f"{lic}.txt").write_text(f"text of {lic}\n", encoding="utf-8")
    sha = {lic: hashlib.sha256(f"text of {lic}\n".encode()).hexdigest() for lic in ("ok", "pending")}
    (tmp_path / LICENSES_LOCK).write_text(
        "# header | is | ignored\n"
        f"ok | url | {sha['ok']} | raw | x | 2026-10-13 | research | bao-security APPROVED 2026-10-13 (re-fetched)\n"
        f"pending | url | {sha['pending']} | raw | x | 2026-10-13 | research | pending bao-security\n"
        f"tampered | url | {'0' * 64} | raw | x | 2026-10-13 | research | bao-security APPROVED\n",
        encoding="utf-8",
    )
    assert license_text(tmp_path, "ok") == tmp_path / LICENSES_DIR / "ok.txt"
    assert license_text(tmp_path, "pending") is None  # not approved
    assert license_text(tmp_path, "tampered") is None  # sha256 mismatch
    assert license_text(tmp_path, "absent") is None


@pytest.mark.parametrize(
    ("verdict", "ok"),
    [
        ("bao-security APPROVED", True),
        ("bao-security APPROVED 2026-10-13 (re-fetched)", True),
        ("bao-security APPROVED(2026-10-13)", True),
        ("bao-security APPROVED-PENDING", False),
        ("APPROVED", False),  # whose approval?
        ("someone APPROVED", False),
        ("bao-security approved", False),
        ("PENDING bao-security", False),
    ],
)
def test_one_security_mark_for_every_lock(verdict: str, ok: bool) -> None:
    from vlm_pipeline.common.approval import is_security_approved

    assert is_security_approved(verdict) is ok
