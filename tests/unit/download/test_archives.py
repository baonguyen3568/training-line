"""P021 H5b / security checklist §10: WebDataset shards are streamed under rules; any bad member refuses the shard."""

from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlm_pipeline.download import archives as ar


def _tar(path: Path, members: list[tuple[str, bytes | None, bytes]], *, gz: bool = False) -> Path:
    """members: (name, data or None, type) with tarfile type codes."""
    with tarfile.open(path, "w:gz" if gz else "w") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "a.jpg"
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)
    return path


def _samples(path: Path) -> tuple[list[dict[str, Any]], ar.ShardReport]:
    report = ar.ShardReport()
    return list(ar.iter_samples([path], report)), report


def test_samples_are_grouped_by_key_and_fields(tmp_path: Path) -> None:
    reg = tarfile.REGTYPE
    members = [("d/s1.jpg", b"J1", reg), ("d/s1.json", b'{"q": 1}', reg), ("d/s2.seg.png", b"P2", reg),
               ("d/s2.txt", b"t", reg), ("d/s3.npy", b"x", reg), ("d/s3.tar", b"x", reg)]  # fmt: skip
    shard = _tar(tmp_path / "s.tar", members)
    samples, report = _samples(shard)
    assert samples == [{"__key__": "d/s1", "jpg": b"J1", "json": b'{"q": 1}'},
                       {"__key__": "d/s2", "seg.png": b"P2", "txt": b"t"}]  # fmt: skip
    assert dict(report.skipped) == {"extension": 1, "nested_archive": 1}  # never opened, never written


@pytest.mark.parametrize(
    ("member", "reason"),
    [
        (("../x.jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("/abs.jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("a\\b.jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("C:x.jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("a\x01.jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("x" * 300 + ".jpg", b"x", tarfile.REGTYPE), "bad_name"),
        (("l.jpg", None, tarfile.SYMTYPE), "not_a_regular_member"),
        (("h.jpg", None, tarfile.LNKTYPE), "not_a_regular_member"),
        (("dev", None, tarfile.CHRTYPE), "not_a_regular_member"),
        (("fifo", None, tarfile.FIFOTYPE), "not_a_regular_member"),
    ],
)
def test_one_bad_member_refuses_the_whole_shard(tmp_path: Path, member: tuple[str, bytes | None, bytes],
                                                reason: str) -> None:  # fmt: skip
    shard = _tar(tmp_path / "s.tar", [("a.jpg", b"ok", tarfile.REGTYPE), member])
    with pytest.raises(ar.ArchiveRefused) as info:
        _samples(shard)
    assert info.value.reason == reason


@pytest.mark.parametrize("second", ["a.jpg", "A.JPG", "á.jpg"])
def test_duplicates_also_case_insensitive_and_after_nfc_are_refused(tmp_path: Path, second: str) -> None:
    first = "a.jpg" if second != "á.jpg" else "á.jpg"  # á precomposed vs a + combining accent
    shard = _tar(tmp_path / "s.tar", [(first, b"1", tarfile.REGTYPE), (second, b"2", tarfile.REGTYPE)])
    with pytest.raises(ar.ArchiveRefused, match="duplicate_name"):
        _samples(shard)


def test_an_oversized_image_member_is_skipped_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar, "MAX_IMAGE", 4)
    shard = _tar(tmp_path / "s.tar", [("a.jpg", b"123456789", tarfile.REGTYPE), ("a.txt", b"t", tarfile.REGTYPE)])
    samples, report = _samples(shard)
    assert samples == [{"__key__": "a", "txt": b"t"}]
    assert report.skipped["too_large"] == 1


def test_a_gzip_bomb_is_refused_while_reading(tmp_path: Path) -> None:
    shard = _tar(tmp_path / "b.tar.gz", [(f"s{n}.txt", b"\0" * (900 << 10), tarfile.REGTYPE) for n in range(60)],
                 gz=True)  # fmt: skip
    with pytest.raises(ar.ArchiveRefused, match="compression_bomb"):
        _samples(shard)


def _zip(path: Path, members: list[tuple[str, bytes]], *, mode: int | None = None) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, data)
    return path


def test_a_zip_shard_streams_like_a_tar(tmp_path: Path) -> None:
    samples, _ = _samples(_zip(tmp_path / "s.zip", [("s1.jpg", b"J"), ("s1.txt", b"T")]))
    assert samples == [{"__key__": "s1", "jpg": b"J", "txt": b"T"}]


def test_a_42_zip_style_ratio_bomb_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ar.ArchiveRefused, match="compression_bomb"):
        _samples(_zip(tmp_path / "b.zip", [("big.txt", b"\0" * (2 << 20))]))


def test_an_encrypted_zip_is_refused(tmp_path: Path) -> None:
    path = _zip(tmp_path / "e.zip", [("a.txt", b"t")])
    raw = bytearray(path.read_bytes())  # zipfile never writes the flag itself: set it in both headers
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        at = raw.index(signature) + offset
        raw[at] |= 0x1
    path.write_bytes(bytes(raw))
    with pytest.raises(ar.ArchiveRefused, match="encrypted"):
        _samples(path)


def test_a_zip_symlink_member_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ar.ArchiveRefused, match="not_a_regular_member"):
        _samples(_zip(tmp_path / "l.zip", [("a.jpg", b"target")], mode=0o120777))


def test_overlapping_zip_entries_are_refused() -> None:
    infos = [SimpleNamespace(header_offset=0, filename="a.txt", extra=b"", compress_size=100, file_size=100,
                             flag_bits=0, volume=0),
             SimpleNamespace(header_offset=50, filename="b.txt", extra=b"", compress_size=10, file_size=10,
                             flag_bits=0, volume=0)]  # fmt: skip
    with pytest.raises(ar.ArchiveRefused, match="overlapping_entries"):
        ar._zip_layout_ok(SimpleNamespace(), infos, 1000)  # type: ignore[arg-type]


def test_a_nested_or_unknown_archive_is_not_a_shard(tmp_path: Path) -> None:
    (tmp_path / "s.7z").write_bytes(b"7z")
    with pytest.raises(ar.ArchiveRefused, match="unsupported_archive"):
        _samples(tmp_path / "s.7z")
    plain = tmp_path / "p.tar"
    plain.write_bytes(gzip.compress(b"not a tar"))
    with pytest.raises(ar.ArchiveRefused, match="unreadable"):
        _samples(plain)


# ------------------------------------------------------------------------------------------- the pipeline (H5b)
def _jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (0, 0, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _wds(path: Path, n: int = 3) -> Path:
    import json

    members = []
    for i in range(n):
        text = json.dumps({"question": f"Q{i}?", "answer": f"A{i}"}).encode()
        members += [(f"s{i}.jpg", _jpeg(), tarfile.REGTYPE), (f"s{i}.json", text, tarfile.REGTYPE)]
    return _tar(path, members)


def test_shards_are_a_webdataset_split_and_only_user_datasets_may_fetch_them() -> None:
    from vlm_pipeline.common.download_rules import wanted
    from vlm_pipeline.download.dataset_layout import layout

    tree = layout(["train/docvqa-train-0000.tar", "train/docvqa-train-0001.tar", "val/docvqa-val-0000.tar"])
    assert {s: [f.format for f in fs] for s, fs in tree["default"].items()} == {
        "train": ["webdataset", "webdataset"], "validation": ["webdataset"]}  # fmt: skip
    assert not wanted("a.tar", "dataset")  # a catalog download: never an archive
    assert wanted("a.tar", "dataset", archives=True)
    assert not wanted("a.tar", "model", archives=True)
    assert not wanted("a.7z", "dataset", archives=True)


def test_inspect_reads_a_shard_into_columns(tmp_path: Path) -> None:
    from vlm_pipeline.download import inspect as ins

    sample = tmp_path / "sample"
    sample.mkdir()
    _wds(sample / "train-0000.tar")
    result = ins.inspect(sample, "webdataset", tmp_path / "out")
    assert result["suggested"] == {"image": "jpg", "question": "json.question", "answer": "json.answer"}
    assert result["samples"][0]["thumb"] == 0


def test_convert_turns_shards_into_a_checked_local_folder(tmp_path: Path) -> None:
    import json

    from vlm_pipeline.data.local_folder import scan_folder
    from vlm_pipeline.download import convert as cv

    files = tmp_path / "files"
    files.mkdir()
    _wds(files / "train-0000.tar", 2)
    _wds(files / "train-0001.tar", 2)
    spec = {"files": ["train-0000.tar", "train-0001.tar"], "format": "webdataset", "limit": 0, "max_bytes": 1 << 30,
            "mapping": {"image": "jpg", "question": "json.question", "answer": "json.answer"}}  # fmt: skip
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    out.mkdir()
    assert cv.main(["--files", str(files), "--spec", str(tmp_path / "spec.json"), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["rows"] == 4
    scan = scan_folder(out)
    assert (scan.problem_count, len(scan.records)) == (0, 4)


def test_a_refused_shard_stops_the_conversion(tmp_path: Path) -> None:
    import json

    from vlm_pipeline.download import convert as cv

    files = tmp_path / "files"
    files.mkdir()
    _tar(files / "bad.tar", [("../x.jpg", _jpeg(), tarfile.REGTYPE)])
    spec = {"files": ["bad.tar"], "format": "webdataset", "limit": 0, "max_bytes": 1 << 30,
            "mapping": {"image": "jpg", "answer": "txt"}}  # fmt: skip
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "out"
    out.mkdir()
    assert cv.main(["--files", str(files), "--spec", str(tmp_path / "spec.json"), "--out", str(out)]) == 4
    assert json.loads((out / "report.json").read_text()) == {"error": "shard_bad_name"}


def test_a_bomb_hidden_in_a_skipped_member_is_refused_fast(tmp_path: Path) -> None:
    """security H5b B1: a .bin member (skipped) still decompresses; 2 GB of zeros must be refused within a second."""
    import time

    path = tmp_path / "pad.tar.gz"
    info = tarfile.TarInfo("pad.bin")
    info.size = 2 << 30  # 2 GiB of zeros, declared and really there once decompressed
    block = gzip.compress(bytes(1 << 20))  # gzip members concatenate: 2 048 x 1 MiB of zeros, built instantly
    path.write_bytes(gzip.compress(info.tobuf()) + block * 2048 + gzip.compress(bytes(1024)))
    start = time.monotonic()
    with pytest.raises(ar.ArchiveRefused, match="compression_bomb"):
        _samples(path)
    assert time.monotonic() - start < 1.0


def test_a_large_incompressible_first_member_is_not_a_bomb(tmp_path: Path) -> None:
    import os

    noise = os.urandom(3 << 20)  # a real photo does not compress: ratio ~1, never refused
    samples, _ = _samples(_tar(tmp_path / "n.tar.gz", [("a.jpg", noise, tarfile.REGTYPE)], gz=True))
    assert samples[0]["jpg"] == noise


def test_declared_sizes_count_towards_the_shard_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar, "MAX_SHARD_BYTES", 10)
    shard = _tar(tmp_path / "s.tar", [("a.bin", b"x" * 20, tarfile.REGTYPE)])  # skipped, still counted
    with pytest.raises(ar.ArchiveRefused, match="shard_too_large"):
        _samples(shard)
