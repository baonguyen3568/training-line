"""Proposal 012: lineage, fail-closed licence texts, and the files an export carries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vlm_pipeline.common.catalog import load_catalog, load_dataset_catalog
from vlm_pipeline.common.licenses import LICENSES_DIR, LICENSES_LOCK, license_text
from vlm_pipeline.common.paths import project_root
from vlm_pipeline.common.runinfo import RunManifest, write_manifest
from vlm_pipeline.export import notices

PARSE = next(m for m in load_catalog().models if m.id == "nemotron-parse-v1.1")
CORD = next(d for d in load_dataset_catalog().datasets if d.repo == "naver-clova-ix/cord-v2")


def _run(root: Path, run_id: str, **kw: object) -> Path:
    run_dir = root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_manifest(run_dir, RunManifest(run_id=run_id, name=run_id, stage="train", task="sft", **kw))
    return run_dir


def _vendor(root: Path, *ids: str, verdict: str = "bao-security APPROVED 2026-10-13") -> None:
    d = root / LICENSES_DIR
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for lic in ids:
        text = f"VERBATIM TEXT OF {lic}\n".encode()
        (d / f"{lic}.txt").write_bytes(text)
        rows.append(
            f"{lic} | https://example.invalid/{lic} | {hashlib.sha256(text).hexdigest()} | 2026-10-13 | t | {verdict}"
        )
    (root / LICENSES_LOCK).write_text("# id | url | sha256 | fetched_at | fetched_by | verdict\n" + "\n".join(rows))


def _lineage(root: Path, artifact: Path, base_ref: str | None = None):  # type: ignore[no-untyped-def]
    return notices.lineage(
        artifact, root=root, catalog=load_catalog(), datasets=load_dataset_catalog(), base_ref=base_ref
    )


def test_trained_artifact_lists_base_model_and_training_data(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", base_model=PARSE.repo, base_model_revision=PARSE.revision,
               dataset=f"{CORD.repo}@{CORD.revision}#limit=100", ended_at=0.0)  # fmt: skip
    artifact = run / "checkpoints" / "LATEST" / "model" / "consolidated"
    artifact.mkdir(parents=True)
    components, mods, gaps = _lineage(tmp_path, artifact)
    assert [(c.role, c.repo, c.license) for c in components] == [
        ("base_model", PARSE.repo, PARSE.license),
        ("training_data", CORD.repo, CORD.license),
    ]
    assert components[1].revision == CORD.revision  # "#limit=" is not part of the revision
    assert [m.run_id for m in mods] == ["r1"]
    assert gaps == []


def test_parent_runs_are_followed_and_shared_data_listed_once(tmp_path: Path) -> None:
    _run(
        tmp_path,
        "r0",
        base_model=PARSE.repo,
        base_model_revision=PARSE.revision,
        dataset=f"{CORD.repo}@{CORD.revision}",
    )
    child = _run(tmp_path, "r1", parent_run_id="r0", dataset=f"{CORD.repo}@{CORD.revision}")
    components, mods, gaps = _lineage(tmp_path, child)
    assert [c.role for c in components] == ["base_model", "training_data"]
    assert [m.run_id for m in mods] == ["r1", "r0"]
    assert gaps == []


def test_untrained_catalog_model_and_unknown_data_are_reported_not_guessed(tmp_path: Path) -> None:
    components, mods, gaps = _lineage(tmp_path, tmp_path / "snapshot", base_ref=f"{PARSE.repo}@{PARSE.revision}")
    assert [c.role for c in components] == ["base_model"]
    assert (mods, gaps) == ([], [])
    run = _run(tmp_path, "r2", base_model="someone/unknown", base_model_revision="0" * 40, dataset="/data/mine")
    components, _, gaps = _lineage(tmp_path, run)
    assert [(c.role, c.license) for c in components] == [("training_data", None)]
    assert gaps == [
        f"base model someone/unknown@{'0' * 40} is not a catalog model",
        "training data local data 'mine' (path not recorded): licence not recorded",
    ]


def test_missing_parent_is_a_gap(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", parent_run_id="gone", dataset=f"{CORD.repo}@{CORD.revision}")
    _, _, gaps = _lineage(tmp_path, run)
    assert "parent run gone not found" in gaps[0]


def test_licence_text_must_match_the_lock(tmp_path: Path) -> None:
    _vendor(tmp_path, "cc-by-4.0")
    assert license_text(tmp_path, "cc-by-4.0") is not None
    (tmp_path / LICENSES_DIR / "cc-by-4.0.txt").write_text("edited by hand\n")
    assert license_text(tmp_path, "cc-by-4.0") is None
    assert license_text(tmp_path, "apache-2.0") is None


def test_every_catalog_licence_has_an_approved_vendored_text() -> None:
    """The real configs/licenses: an export of any catalog model or dataset can be written."""
    root = project_root()
    ids = {m.license for m in load_catalog().models} | {d.license for d in load_dataset_catalog().datasets}
    assert {i for i in ids if license_text(root, i) is None} == set()


def test_notice_placeholders_are_filled_and_unknown_ones_stay_visible() -> None:
    template = "Based on {repo}@{revision}. Modified: fine-tuned with {tool} on {date}. {oops}"
    c = notices.Component(role="base_model", name="Q", repo="org/q", revision="abc", license="apache-2.0",
                          notices=(template,))  # fmt: skip
    [filled] = notices.fill_notices([c], tool="VLM Studio", date="2026-10-13")
    expected = "Based on org/q@abc. Modified: fine-tuned with VLM Studio on 2026-10-13. {oops}"
    assert filled.notices == (expected,)


def test_export_refuses_without_locked_texts_and_writes_nothing(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        "r1",
        base_model=PARSE.repo,
        base_model_revision=PARSE.revision,
        dataset=f"{CORD.repo}@{CORD.revision}",
    )
    components, mods, gaps = _lineage(tmp_path, run)
    _vendor(tmp_path, CORD.license)  # the model licence is missing
    dest = tmp_path / "export"
    dest.mkdir()
    with pytest.raises(notices.LicenseTextMissingError) as ei:
        notices.write_notices(
            dest, root=tmp_path, components=components, modifications=mods, gaps=gaps, catalog=load_catalog()
        )
    assert ei.value.licenses == [PARSE.license]
    assert list(dest.iterdir()) == []


def test_export_writes_licences_notice_card_and_hashes(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", base_model=PARSE.repo, base_model_revision=PARSE.revision,
               dataset=f"{CORD.repo}@{CORD.revision}", ended_at=0.0)  # fmt: skip
    components, mods, gaps = _lineage(tmp_path, run)
    _vendor(tmp_path, PARSE.license, CORD.license)
    dest = tmp_path / "export"
    dest.mkdir()
    (dest / "model.safetensors").write_bytes(b"weights")
    upstream = tmp_path / "snapshot"
    upstream.mkdir()
    (upstream / "NOTICE").write_text("upstream notice\n")
    att = notices.write_notices(
        dest, root=tmp_path, components=components, modifications=mods, gaps=gaps, catalog=load_catalog(),
        upstream_dir=upstream,
    )  # fmt: skip
    assert (dest / "NOTICE.upstream").read_text() == "upstream notice\n"
    assert (dest / "LICENSE").read_text() == f"VERBATIM TEXT OF {PARSE.license}\n"
    assert (dest / "LICENSES" / f"{CORD.license}.txt").is_file()
    notice = (dest / "NOTICE").read_text()
    for needle in (PARSE.display_name, PARSE.revision, CORD.repo, CORD.attribution, "sft run r1"):
        assert needle in notice
    card = (dest / "README.md").read_text()
    assert PARSE.repo in card
    assert "Known limitations" in card
    on_disk = json.loads((dest / "ATTRIBUTION.json").read_text())
    assert on_disk["schema_version"] == 1
    assert on_disk == att.model_dump(mode="json")
    assert att.files["model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert set(att.files) == {
        "model.safetensors", "LICENSE", f"LICENSES/{CORD.license}.txt", "NOTICE", "NOTICE.upstream", "README.md",
    }  # fmt: skip


def test_missing_base_revision_is_a_gap_not_a_match(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", base_model=PARSE.repo, dataset=f"{CORD.repo}@{CORD.revision}")
    components, _, gaps = _lineage(tmp_path, run)
    assert [c.role for c in components] == ["training_data"]
    assert gaps == [f"base model {PARSE.repo}: revision not recorded"]


def test_local_data_path_never_reaches_the_export(tmp_path: Path) -> None:
    run = _run(tmp_path, "r1", base_model=PARSE.repo, base_model_revision=PARSE.revision,
               dataset="/home/someone/private/invoices-2026#limit=10")  # fmt: skip
    components, mods, gaps = _lineage(tmp_path, run)
    shared = json.dumps([c.model_dump() for c in components] + [m.model_dump() for m in mods]) + " ".join(gaps)
    assert "/home/someone" not in shared
    assert "local data 'invoices-2026' (path not recorded)" in shared
    assert any("licence not recorded" in g for g in gaps)


def test_usage_enables_remote_code_only_when_the_model_needs_it() -> None:
    native = next(m for m in load_catalog().models if m.remote_code == "none")
    assert not any("trust_remote_code" in line for line in notices._usage(native))
    remote = notices._usage(PARSE)
    assert any("trust_remote_code=True" in line for line in remote)
    assert "source you trust" in remote[0]


def test_stray_braces_in_a_notice_never_break_the_export() -> None:
    c = notices.Component(role="base_model", name="Q", repo="org/q", revision="abc", license="x",
                          notices=("Weights {unbalanced and {repo} at {revision}}",))  # fmt: skip
    [filled] = notices.fill_notices([c], tool="t", date="d")
    assert filled.notices == ("Weights {unbalanced and org/q at abc}",)
