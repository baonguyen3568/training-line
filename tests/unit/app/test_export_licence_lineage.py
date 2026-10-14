"""ADR-021 A1 / H5d: a licence-less user model or HF dataset anywhere in an export's lineage refuses the export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.catalog import (
    load_catalog,
    load_dataset_catalog,
    set_user_dataset_lookup,
    user_dataset_id,
)
from vlm_pipeline.common.dataset_ref import DatasetRef
from vlm_pipeline.export import notices

SHA = "c" * 40
DS = user_dataset_id("org/d", SHA, "default", "train")


@pytest.fixture
def datasets() -> Any:
    known: dict[str, dict[str, object]] = {}
    set_user_dataset_lookup(known.get)
    yield known
    set_user_dataset_lookup(None)


def test_origins_are_part_of_the_identity_and_survive_a_split() -> None:
    ref = DatasetRef(path="/w/x", split="all", records="/w/r.jsonl", records_sha256="a" * 64, origins=(DS,))
    assert ref.hash.endswith(f"#origin={DS}")
    assert ref.model_copy(update={"split": "train"}).origins == (DS,)
    plain = ref.model_copy(update={"origins": ()})
    assert plain.hash != ref.hash  # same records from another origin: another dataset


def _train_run(root: Path, dataset: str) -> Path:
    run = root / "outputs" / "graph" / "runs" / "t-1"
    run.mkdir(parents=True)
    body = {"schema_version": 1, "run_id": "t-1", "name": "t", "stage": "train", "task": "train.lora_vlm",
            "status": "completed", "dataset": dataset, "base_model": "Qwen/Qwen3.5-2B",
            "base_model_revision": load_catalog().get("qwen3.5-2b").revision}  # type: ignore[union-attr]  # fmt: skip
    (run / "run.json").write_text(json.dumps(body))
    (run / "model").mkdir()
    return run / "model"


@pytest.mark.parametrize(("status", "licence"), [
    ({"name": "d (default/train)", "repo": "org/d", "revision": SHA, "license": "unknown"}, "unknown"),
    ({"name": "d (default/train)", "repo": "org/d", "revision": SHA, "license": "mit"}, "mit"),
    (None, "unknown"),  # not added here: never presumed fine
])  # fmt: skip
def test_lineage_names_each_user_dataset_and_its_licence(
    tmp_path: Path, datasets: dict[str, Any], status: dict[str, Any] | None, licence: str
) -> None:
    if status is not None:
        datasets[DS] = status
    model = _train_run(tmp_path, f"local:{'a' * 64}#split=train#origin={DS}")
    components, _, _ = notices.lineage(model, root=tmp_path, catalog=load_catalog(), datasets=load_dataset_catalog())
    [data] = [c for c in components if c.role == "training_data"]
    assert data.license == licence
    from vlm_pipeline.app.export_worker import ExportLicenseUnknownError, refuse_unclear_licences

    if licence == "unknown":
        with pytest.raises(ExportLicenseUnknownError) as info:
            refuse_unclear_licences(components, "x")
        assert info.value.code == "SEC-006"
    else:
        refuse_unclear_licences(components, "x")


def test_a_plain_local_folder_is_never_blocked(tmp_path: Path, datasets: dict[str, Any]) -> None:
    """PM + security 2026-10-14: the user's own local data is theirs to declare; only HF origins are checked."""
    model = _train_run(tmp_path, f"local:{'a' * 64}#split=train")
    components, _, _ = notices.lineage(model, root=tmp_path, catalog=load_catalog(), datasets=load_dataset_catalog())
    from vlm_pipeline.app.export_worker import refuse_unclear_licences

    refuse_unclear_licences(components, "x")


def _wf(license_known: bool) -> dict[str, Any]:
    def n(i: str, t: str, **p: Any) -> dict[str, Any]:
        return {"id": i, "type": t, "version": "1.0.0", "params": p}

    return {"format": "vlm-workflow", "schema_version": 1, "name": "t", "nodes": [
        n("m", "model.hf_pick"), n("d", "data.hf_load", user_dataset=DS, user_repo="org/d", user_revision=SHA),
        n("t", "train.lora_vlm"), n("g", "train.merge_lora"), n("x", "export.hf_dir")],
        "edges": [_e("m", "model", "t", "base_model"), _e("d", "dataset", "t", "train_data"),
                  _e("m", "model", "g", "base_model"), _e("t", "adapter", "g", "adapter"),
                  _e("g", "model", "x", "model")]}  # fmt: skip


def _e(a: str, out: str, b: str, port: str) -> dict[str, list[str]]:
    return {"from": [a, out], "to": [b, port]}


@pytest.mark.parametrize(("licence", "flagged"), [("unknown", True), ("cc-by-4.0", False)])
def test_validate_flags_the_export_step(tmp_path: Path, datasets: dict[str, Any], licence: str, flagged: bool) -> None:
    from vlm_pipeline.graph import validate
    from vlm_pipeline.nodes import REGISTRY
    from vlm_pipeline.nodes.rules import export_needs_known_dataset_licences
    from vlm_pipeline.nodes.spec import GraphView

    datasets[DS] = {"name": "d (default/train)", "repo": "org/d", "revision": SHA, "license": licence, "rows": 3,
                    "download": None}  # fmt: skip
    wf = _wf(licence != "unknown")
    specs = {n["id"]: REGISTRY.get(n["type"]) for n in wf["nodes"]}
    params = {n["id"]: specs[n["id"]].validate_params(n["params"]) for n in wf["nodes"]}  # type: ignore[union-attr]
    sources = {(e["to"][0], e["to"][1]): (e["from"][0],) for e in wf["edges"]}
    view = GraphView(specs=specs, params=params, sources=sources, workspace=None)  # type: ignore[arg-type]
    issues = list(export_needs_known_dataset_licences(view))
    assert [(i.code, i.node_id, i.params["repo"]) for i in issues] == ([("SEC-006", "x", "org/d")] if flagged else [])
    codes = {i.code for i in validate(wf, REGISTRY, workspace=tmp_path)}
    assert ("SEC-006" in codes) is flagged


@pytest.mark.parametrize(("card", "seen"), [
    ("Apache-2.0", "apache-2.0"), (" mit ", "mit"), ("other", "unknown"), ("", "unknown"), (None, "unknown"),
    ("llama3", "unknown"), ("nvidia-open-model-license", "nvidia-open-model-license"),
    ("mixed-unverified", "unknown")])  # fmt: skip
def test_only_a_known_licence_id_is_known(card: object, seen: str) -> None:
    """security H5d: a card's "other" (its own terms) or a custom tag is not a licence the app understands."""
    from vlm_pipeline.common.catalog import normalize_license

    assert normalize_license(card) == seen


@pytest.mark.parametrize(("card", "blocked"), [("other", True), ("apache-2.0", False)])
def test_a_card_saying_other_never_exports(tmp_path: Path, card: str, blocked: bool) -> None:
    from vlm_pipeline.app.export_worker import ExportLicenseUnknownError
    from vlm_pipeline.app.workers import _check_export
    from vlm_pipeline.download import resolver
    from vlm_pipeline.graph.worker import SupportQuery
    from vlm_pipeline.nodes import REGISTRY

    family = resolver.load_families().for_architecture("Qwen3_5ForConditionalGeneration")
    assert family is not None
    entry = resolver.user_entry(repo="o/x", revision=SHA, family=family, params_b=1.0, disk_gb=1.0, license_id=card,
                                gated=False)  # fmt: skip
    spec = REGISTRY.get("export.hf_dir")
    assert spec is not None
    query = SupportQuery("x", spec, spec.validate_params({}), entry, {})
    if blocked:
        with pytest.raises(ExportLicenseUnknownError):
            _check_export(query)
    else:
        _check_export(query)


def test_a_dataset_only_compared_against_never_blocks(tmp_path: Path, datasets: dict[str, Any]) -> None:
    """cleancode H5d: dedupe's heldout input is a reference; it never reaches the weights, so its licence is moot."""
    from vlm_pipeline.nodes import REGISTRY
    from vlm_pipeline.nodes.rules import export_needs_known_dataset_licences
    from vlm_pipeline.nodes.spec import GraphView

    other = user_dataset_id("org/other", SHA, "default", "train")
    datasets[other] = {"name": "other", "repo": "org/other", "revision": SHA, "license": "unknown"}
    nodes = {"m": ("model.hf_pick", {}), "d": ("data.hf_load", {}),
             "h": ("data.hf_load", {"user_dataset": other, "user_repo": "org/other", "user_revision": SHA}),
             "u": ("data.dedupe", {}), "t": ("train.lora_vlm", {}), "g": ("train.merge_lora", {}),
             "x": ("export.hf_dir", {})}  # fmt: skip
    specs = {k: REGISTRY.get(t) for k, (t, _) in nodes.items()}
    params = {k: specs[k].validate_params(p) for k, (_, p) in nodes.items()}  # type: ignore[union-attr]
    sources = {("u", "dataset"): ("d",), ("u", "heldout"): ("h",), ("t", "train_data"): ("u",),
               ("t", "base_model"): ("m",), ("g", "base_model"): ("m",), ("g", "adapter"): ("t",),
               ("x", "model"): ("g",)}  # fmt: skip
    view = GraphView(specs=specs, params=params, sources=sources, workspace=None)  # type: ignore[arg-type]
    assert list(export_needs_known_dataset_licences(view)) == []
    sources[("u", "dataset")] = ("h",)  # the same dataset as training data: blocked
    assert [i.code for i in export_needs_known_dataset_licences(view)] == ["SEC-006"]


@pytest.mark.parametrize(("card", "name", "seen"), [
    ("other", "nvidia-open-model-license", "nvidia-open-model-license"),  # nvidia/Cosmos-Reason2-2B
    ("other", "qwen", "qwen"),  # Qwen/Qwen2.5-VL-72B
    ("other", "lfm1.0", "lfm1.0"),  # LiquidAI/LFM2-VL
    ("other", "my-own-terms", "unknown"),
    ("other", None, "unknown"),
    ("apache-2.0", "qwen", "apache-2.0"),  # license_name only fills in for "other"
])  # fmt: skip
def test_other_with_a_known_license_name_is_known(card: str, name: str | None, seen: str) -> None:
    from vlm_pipeline.common.catalog import normalize_license

    assert normalize_license(card, name) == seen


def test_the_resolve_reads_license_name_and_the_record_keeps_the_cards_words(tmp_path: Path) -> None:
    import json as _json

    from vlm_pipeline.app import user_assets

    run = tmp_path / "run"
    meta = run / "stage" / "meta"
    meta.mkdir(parents=True)
    (meta / "config.json").write_text(_json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                   "model_type": "qwen3_5"}))  # fmt: skip
    files = [{"name": "config.json", "size": 10, "sha256": None, "git_sha1": None},
             {"name": "model.safetensors", "size": 2_000_000_000, "sha256": None, "git_sha1": None}]  # fmt: skip
    (run / "stage" / "resolve.json").write_text(_json.dumps({
        "sha": SHA, "gated": False, "license": "other", "license_name": "qwen", "all_files": ["config.json",
        "model.safetensors"], "files": files, "total_bytes": 2_000_000_010, "unreadable": []}))  # fmt: skip
    record = user_assets.finish_model(tmp_path, run, "Qwen/x")
    assert (record.entry.license, record.license_card) == ("qwen", "other (qwen)")
