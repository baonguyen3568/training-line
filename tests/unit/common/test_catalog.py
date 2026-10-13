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
    from vlm_pipeline.common.catalog import license_ids_needing_text
    from vlm_pipeline.common.licenses import license_text
    from vlm_pipeline.common.paths import project_root

    ids = license_ids_needing_text(load_catalog(), load_dataset_catalog())
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


@pytest.mark.parametrize(
    ("remote_code", "auto_class", "ok"),
    [
        ("pinned_prepared", "AutoModel", True),
        ("none", "AutoModelForImageTextToText", True),
        ("none", "AutoModel", False),  # AutoModel resolves only through the repo's auto_map
        ("pinned_prepared", "AutoModelForImageTextToText", False),
    ],
)
def test_auto_class_matches_remote_code(remote_code: str, auto_class: str, ok: bool) -> None:
    raw = _model(remote_code=remote_code, auto_class=auto_class)
    if ok:
        CatalogEntry.model_validate(raw)
    else:
        with pytest.raises(ValidationError, match="auto_class"):
            CatalogEntry.model_validate(raw)


def _template_model_ids() -> dict[str, set[str]]:
    """Catalog model id -> ids of the shipped templates whose model.hf_pick uses it (no param = catalog default)."""
    from vlm_pipeline.app.templates import load_templates
    from vlm_pipeline.common.paths import project_root

    used: dict[str, set[str]] = {}
    for template in load_templates(project_root()):
        for node in template["workflow"]["nodes"]:
            if node["type"] == "model.hf_pick":
                model = node.get("params", {}).get("model", load_catalog().default.id)
                used.setdefault(model, set()).add(template["id"])
    return used


def test_templates_use_template_tier_models_and_every_template_tier_model_is_used() -> None:
    tiers = {m.id: m.tier for m in load_catalog().models}
    used = _template_model_ids()
    wrong = {m: sorted(t) for m, t in used.items() if tiers.get(m) not in ("default", "template")}
    assert wrong == {}, f"templates use models whose catalog tier is not default/template: {wrong}"
    unused = sorted(m for m, tier in tiers.items() if tier == "template" and m not in used)
    assert unused == [], f"tier: template but no shipped template uses it: {unused}"


@pytest.mark.parametrize(
    ("override", "ok"),
    [
        (
            {
                "license": "mixed-unverified",
                "use": "eval_only",
                "commercial": "conditional",
                "files_sha256": {"data/t.parquet": "a" * 64},
            },
            True,
        ),
        ({"license": "mixed-unverified", "use": "eval_only", "commercial": "conditional"}, False),  # hashes required
        ({"license": "mixed-unverified", "use": "train"}, False),  # cannot train on unverified licences
        ({"license": "mixed-unverified", "use": "eval_only", "commercial": "yes"}, False),
        ({"files_sha256": {"data/test-00000-of-00001.parquet": "a" * 64}}, True),
        ({"files_sha256": {"../escape.parquet": "a" * 64}}, False),
        ({"files_sha256": {"/abs.parquet": "a" * 64}}, False),
        ({"files_sha256": {"data/x.parquet": "not-a-sha"}}, False),
    ],
)
def test_dataset_licence_and_file_hashes(override: dict[str, Any], ok: bool) -> None:
    raw = {**load_dataset_catalog().datasets[0].model_dump(), **override}
    if override.get("use") == "eval_only":
        raw["heldout_splits"] = list(raw["splits"])
    if ok:
        DatasetEntry.model_validate(raw)
    else:
        with pytest.raises(ValidationError):
            DatasetEntry.model_validate(raw)


def test_unverified_licence_datasets_are_not_licence_text_requirements() -> None:
    from vlm_pipeline.common.catalog import UNVERIFIED_LICENSE, license_ids_needing_text

    assert UNVERIFIED_LICENSE not in license_ids_needing_text(load_catalog(), load_dataset_catalog())
