"""Licence duties shown in the G-5 consent dialog come from bao-security's table, per licence AND model part."""

from __future__ import annotations

from vlm_pipeline.common.catalog import load_catalog, load_dataset_catalog
from vlm_pipeline.common.obligations import load_obligations, model_obligations


def test_every_catalog_licence_is_in_the_table() -> None:
    table, catalog = load_obligations(), load_catalog()
    used = {m.license for m in catalog.models} | {p.license for m in catalog.models for p in m.model_parts}
    used |= {d.license for d in load_dataset_catalog().datasets}
    assert used - set(table.licences) == set(), (
        "ask bao-security to add the licence to configs/licenses/obligations.yaml"
    )


def test_model_parts_keep_their_own_duties() -> None:
    license_class, duties = model_obligations(load_catalog().get("nemotron-parse-v1.1"), load_obligations())  # type: ignore[arg-type]
    keys = {(d["licence"], d["id"]) for d in duties}
    assert license_class == "conditional"
    assert ("nvidia-open-model-license", "attribution_notice") in keys
    assert ("cc-by-4.0", "cc_by_attribution") in keys  # tokenizer part
    assert len(keys) == len(duties)  # de-duplicated on (licence, id), never on id alone


def test_cosmos_notice_only_for_cosmos_models() -> None:
    table, catalog = load_obligations(), load_catalog()
    for model_id, expected in (("cosmos-reason2-2b", True), ("nemotron-parse-v1.1", False)):
        _, duties = model_obligations(catalog.get(model_id), table)  # type: ignore[arg-type]
        assert any(d["id"] == "built_on_cosmos" for d in duties) is expected, model_id


def test_unknown_terms_are_stricter_than_non_commercial() -> None:
    from vlm_pipeline.common.catalog import ModelPart
    from vlm_pipeline.common.obligations import LicenceTerms, ObligationsTable

    table = ObligationsTable(
        schema_version=1,
        licences={"nc": LicenceTerms(license_class="non_commercial", obligations=())},
        unknown_licence=LicenceTerms(license_class="unknown", obligations=()),
    )
    base = load_catalog().get("qwen3.5-2b")
    assert base is not None
    entry = base.model_copy(update={"license": "nc", "model_parts": (ModelPart(name="vision", license="not-listed"),)})
    assert model_obligations(entry, table)[0] == "unknown"
