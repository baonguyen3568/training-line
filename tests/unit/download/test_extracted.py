"""G-5 eval data (tester 2026-10-13): the host checks a job's extracted tree before it reaches data/eval/."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vlm_pipeline.download import extracted
from vlm_pipeline.download.extracted import (
    MANIFEST,
    ExtractSpec,
    check_source,
    eval_data_dir,
    extracted_ok,
    install_extracted,
    verify_tree,
)
from vlm_pipeline.download.install import DownloadIntegrityError, snapshot_dir

GOOD = {"images/0000.jpg": b"\xff\xd8\xff a", "questions.jsonl": b'{"row": 0}\n'}


def make_tree(path: Path, files: dict[str, bytes] | None = None, manifest_extra: str = "") -> Path:
    files = files if files is not None else GOOD
    for rel, data in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(data)
    lines = [f"{hashlib.sha256(data).hexdigest()}  {rel}" for rel, data in sorted(files.items())]
    (path / MANIFEST).write_text("\n".join(lines) + "\n" + manifest_extra)
    return path


def _manifest_sha(files: dict[str, bytes]) -> str:
    lines = [f"{hashlib.sha256(data).hexdigest()}  {rel}" for rel, data in sorted(files.items())]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> str:
    """The catalog pins the real OCRBench manifest; these tests pin the GOOD tree's instead."""
    real = ExtractSpec.entry
    sha = _manifest_sha(GOOD)

    def entry(self: ExtractSpec) -> object:
        return real(self).model_copy(update={"extracted_manifest_sha256": sha})

    monkeypatch.setattr(extracted.ExtractSpec, "entry", entry)
    return sha


def test_the_catalog_pins_the_real_extraction() -> None:
    from vlm_pipeline.common.catalog import load_dataset_catalog

    entry = load_dataset_catalog().get("ocrbench")
    assert entry is not None
    assert entry.extracted_manifest_sha256 == "e6e6f4460c621bb515ef81744f85dd854a33d75599041b3c6028f04039f29a47"


def test_a_good_tree_is_installed_once(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "out")
    target = install_extracted(tmp_path, "ocrbench", tree)
    assert target == eval_data_dir(tmp_path, "ocrbench")
    assert target.name == "ocrbench-92a54bd1"
    assert (target / "images" / "0000.jpg").read_bytes() == b"\xff\xd8\xff a"
    assert extracted_ok(tmp_path, "ocrbench")
    again = make_tree(tmp_path / "out2")
    assert install_extracted(tmp_path, "ocrbench", again) == target  # kept as is


def _bad(tmp_path: Path, how: str) -> Path:
    tree = make_tree(tmp_path / "out")
    if how == "unlisted file":
        (tree / "images" / "0001.jpg").write_bytes(b"x")
    elif how == "changed file":
        (tree / "questions.jsonl").write_bytes(b"changed")
    elif how == "link":
        (tree / "images" / "0000.jpg").unlink()
        (tree / "images" / "0000.jpg").symlink_to("/etc/hostname")
    elif how == "bad name":
        make_tree(tree, {"images/0000.jpg": b"\xff\xd8\xff a", "questions.jsonl": b"{}\n", "run.sh": b"rm"})
    elif how == "gif":
        make_tree(tree, {"images/0000.gif": b"GIF89a", "questions.jsonl": b"{}\n"})
    elif how == "no questions":
        (tree / "questions.jsonl").unlink()
        make_tree(tree, {"images/0000.jpg": b"\xff\xd8\xff a"})
    elif how == "manifest link":
        (tree / MANIFEST).rename(tmp_path / "m")
        (tree / MANIFEST).symlink_to(tmp_path / "m")
    return tree


@pytest.mark.parametrize(
    "how", ["unlisted file", "changed file", "link", "bad name", "gif", "no questions", "manifest link"]
)
def test_anything_off_installs_nothing(tmp_path: Path, how: str) -> None:
    tree = _bad(tmp_path, how)
    with pytest.raises(DownloadIntegrityError):
        install_extracted(tmp_path, "ocrbench", tree)
    assert not eval_data_dir(tmp_path, "ocrbench").exists()
    with pytest.raises(DownloadIntegrityError):
        verify_tree(tree, name="OCRBench", manifest_sha256=_manifest_sha(GOOD))


def test_the_source_must_be_the_catalogs_file(tmp_path: Path) -> None:
    from vlm_pipeline.common.catalog import load_dataset_catalog

    entry = load_dataset_catalog().get("ocrbench")
    assert entry is not None
    parquet = snapshot_dir(tmp_path, "dataset", entry.repo, entry.revision) / "data" / "test-00000-of-00001.parquet"
    with pytest.raises(DownloadIntegrityError):
        check_source(tmp_path, "ocrbench")  # missing
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"not the parquet")
    with pytest.raises(DownloadIntegrityError):
        check_source(tmp_path, "ocrbench")  # wrong bytes


def test_a_self_consistent_but_different_extraction_is_refused(tmp_path: Path) -> None:
    """security: the manifest must be THE known one, not merely match its own tree."""
    tree = make_tree(tmp_path / "out", {"images/0000.jpg": b"\xff\xd8\xff other", "questions.jsonl": b"{}\n"})
    with pytest.raises(DownloadIntegrityError):
        install_extracted(tmp_path, "ocrbench", tree)


def test_extracted_ok_checks_every_file(tmp_path: Path) -> None:
    target = make_tree(eval_data_dir(tmp_path, "ocrbench"))
    assert extracted_ok(tmp_path, "ocrbench")
    (target / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff changed")  # same name, different bytes
    assert not extracted_ok(tmp_path, "ocrbench")


def test_a_damaged_install_is_moved_aside_and_replaced(tmp_path: Path) -> None:
    """tester: a user pressing "Download" on a damaged copy must get a working one, never a dead end."""
    target = make_tree(eval_data_dir(tmp_path, "ocrbench"))
    (target / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff changed")
    install_extracted(tmp_path, "ocrbench", make_tree(tmp_path / "out"))
    assert extracted_ok(tmp_path, "ocrbench")
    [aside] = [p for p in target.parent.iterdir() if p.name.startswith(f".{target.name}.invalid-")]
    assert (aside / "images" / "0000.jpg").read_bytes() == b"\xff\xd8\xff changed"  # kept, never deleted
