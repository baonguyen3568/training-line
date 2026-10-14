r"""Read WebDataset shards (``.tar`` / ``.tar.gz`` / ``.zip``) member by member, never extracting (P021 H5b).

Security checklist 2026-10-14 §10, "do not extract when you can stream": each regular member is checked and handed
to the caller as bytes; nothing from an archive is ever written under its own name. Any of these refuses the WHOLE
shard (``ArchiveRefused(reason)``), keeping nothing of it:

* a symlink / hardlink / device / fifo / sparse or contiguous tar member, an encrypted or multi-disk zip, a zip whose
  directory and local headers disagree or whose entries overlap;
* a name that is absolute, has ``..`` / ``\`` / a drive letter / a control character, a component over 255 bytes,
  a path over 512, or a duplicate (also case-insensitive after NFC);
* over 200 000 members or 8 GiB uncompressed in one shard; a compression ratio over 20 (gzip tar, zip total) or 100
  (one zip member) - a bomb.

Members with another extension are skipped and counted; an archive inside an archive is never opened (skipped,
counted). Per-member caps: 20 MiB image, 16 MiB json/jsonl, 1 MiB other text.
"""

from __future__ import annotations

import itertools
import posixpath
import re
import tarfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, cast

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
JSON_EXT = (".json", ".jsonl")
TEXT_EXT = (".txt", ".cls", ".caption", ".text")
ARCHIVE_EXT = (".tar", ".tgz", ".tar.gz", ".zip", ".7z", ".rar", ".gz", ".xz", ".zst", ".bz2")
MAX_MEMBERS = 200_000
MAX_SHARD_BYTES = 8 << 30
MAX_IMAGE = 20 << 20
MAX_JSON = 16 << 20
MAX_TEXT = 1 << 20
MAX_NAME = 512
MAX_COMPONENT = 255
MAX_RATIO = 20  # gzip tar out/in, zip total
MAX_MEMBER_RATIO = 100  # one zip member
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE = re.compile(r"^[A-Za-z]:")


class ArchiveRefused(ValueError):  # noqa: N818 - a refusal with its reason, like ImageRuleError
    """The whole shard is refused; ``reason`` names the rule (never the member's content)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ShardReport:
    skipped: Counter[str] = field(default_factory=Counter)
    members: int = 0
    bytes_out: int = 0  # bytes of the kept members
    declared_out: int = 0  # every member's declared size, kept or skipped (a skipped one is still decompressed)


def check_name(name: str, seen: set[str]) -> str:
    """The member name, NFC-normalised, if it is a safe relative POSIX path not seen before; else refuse."""
    if not name or name.startswith("/") or "\\" in name or _DRIVE.match(name) or _CONTROL.search(name):
        raise ArchiveRefused("bad_name")
    norm = unicodedata.normalize("NFC", name)
    parts = norm.split("/")
    if ".." in parts or len(norm.encode()) > MAX_NAME or any(len(p.encode()) > MAX_COMPONENT for p in parts):
        raise ArchiveRefused("bad_name")
    folded = norm.casefold()
    if folded in seen:
        raise ArchiveRefused("duplicate_name")
    seen.add(folded)
    return norm


def _cap(name: str) -> int | None:
    """The size cap of a kept member, None when its extension is not kept."""
    lower = name.lower()
    if lower.endswith(IMAGE_EXT):
        return MAX_IMAGE
    if lower.endswith(JSON_EXT):
        return MAX_JSON
    if lower.endswith(TEXT_EXT):
        return MAX_TEXT
    return None


class _CountingReader:
    """A file wrapper that counts the compressed bytes read from the shard."""

    def __init__(self, raw: IO[bytes]) -> None:
        self.raw, self.read_bytes = raw, 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.read_bytes += len(data)
        return data


class _RatioReader:
    """The decompressed gzip stream tarfile reads, checked on EVERY byte it produces.

    A member tarfile only skips is still decompressed, so the ratio is measured on the output, not on the kept
    members (security H5b B1).
    """

    def __init__(self, compressed: _CountingReader) -> None:
        import gzip

        self.compressed, self.out = compressed, 0
        self.gz: Any = gzip.GzipFile(fileobj=cast("IO[bytes]", compressed), mode="rb")

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0 or size > (1 << 20):  # never a whole-stream read before the ratio check
            size = 1 << 20
        data: bytes = self.gz.read(size)
        self.out += len(data)
        if self.out > MAX_RATIO * self.compressed.read_bytes + (1 << 20):
            raise ArchiveRefused("compression_bomb")
        return data


def iter_tar(path: Path, report: ShardReport) -> Iterator[tuple[str, bytes]]:
    """(name, bytes) of every kept regular member of a tar / tar.gz shard, streamed in order."""
    seen: set[str] = set()
    gz = path.name.lower().endswith((".tar.gz", ".tgz"))
    with path.open("rb") as raw:
        counter = _CountingReader(raw)
        source: Any = _RatioReader(counter) if gz else counter
        try:
            with tarfile.open(fileobj=source, mode="r|") as tar:
                for member in tar:
                    report.members += 1
                    if report.members > MAX_MEMBERS:
                        raise ArchiveRefused("too_many_members")
                    if member.isdir():
                        continue
                    report.declared_out += max(member.size, 0)  # before it is read or skipped
                    if report.declared_out > MAX_SHARD_BYTES:
                        raise ArchiveRefused("shard_too_large")
                    if not member.isreg() or member.type == tarfile.CONTTYPE or member.issparse():
                        raise ArchiveRefused("not_a_regular_member")  # link, device, fifo, sparse: refuse it all
                    name = check_name(member.name, seen)
                    yield from _kept(tar.extractfile(member), name, member.size, report)
        except (tarfile.TarError, EOFError, OSError) as exc:
            if isinstance(exc, ArchiveRefused):
                raise
            raise ArchiveRefused("unreadable") from None


def _kept(fh: IO[bytes] | None, name: str, size: int, report: ShardReport) -> Iterator[tuple[str, bytes]]:
    if name.lower().endswith(ARCHIVE_EXT):
        report.skipped["nested_archive"] += 1
        return
    cap = _cap(name)
    if cap is None:
        report.skipped["extension"] += 1
        return
    if size > cap or fh is None:
        report.skipped["too_large"] += 1
        return
    data = fh.read(cap + 1)
    if len(data) > cap:
        report.skipped["too_large"] += 1
        return
    report.bytes_out += len(data)
    if report.bytes_out > MAX_SHARD_BYTES:
        raise ArchiveRefused("shard_too_large")
    yield name, data


def iter_zip(path: Path, report: ShardReport) -> Iterator[tuple[str, bytes]]:
    """(name, bytes) of every kept member of a zip shard, after the whole-archive checks."""
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ArchiveRefused("too_many_members")
            _zip_layout_ok(zf, infos, path.stat().st_size)
            for info in infos:
                report.members += 1
                if info.is_dir():
                    continue
                kind = (info.external_attr >> 16) & 0o170000
                if kind not in (0, 0o100000):  # S_IFREG or no unix mode: anything else (link, device...) refuses
                    raise ArchiveRefused("not_a_regular_member")
                if info.compress_size and info.file_size / info.compress_size > MAX_MEMBER_RATIO:
                    raise ArchiveRefused("compression_bomb")
                name = check_name(info.filename, seen)
                with zf.open(info) as fh:  # zipfile itself checks the local header's name against the directory
                    yield from _kept(fh, name, info.file_size, report)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, EOFError, OSError) as exc:
        if isinstance(exc, ArchiveRefused):
            raise
        raise ArchiveRefused("unreadable") from None


def _zip_layout_ok(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], archive_size: int) -> None:
    """Encrypted, multi-disk, overlapping entries (zip quines), or a total ratio over the limit: refused."""
    if any(i.flag_bits & 0x1 for i in infos):
        raise ArchiveRefused("encrypted")
    if getattr(zf, "_start_disk", 0) or any(getattr(i, "volume", 0) for i in infos):
        raise ArchiveRefused("multi_disk")
    ordered = sorted(infos, key=lambda i: i.header_offset)
    for first, second in itertools.pairwise(ordered):
        local_header = 30 + len(first.filename.encode()) + len(first.extra)
        if first.header_offset + local_header + first.compress_size > second.header_offset:
            raise ArchiveRefused("overlapping_entries")
    total_out = sum(i.file_size for i in infos)
    if total_out > MAX_SHARD_BYTES:
        raise ArchiveRefused("shard_too_large")
    if archive_size and total_out > MAX_RATIO * archive_size + (1 << 20):
        raise ArchiveRefused("compression_bomb")


def iter_shard(path: Path, report: ShardReport) -> Iterator[tuple[str, bytes]]:
    lower = path.name.lower()
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        return iter_tar(path, report)
    if lower.endswith(".zip"):
        return iter_zip(path, report)
    raise ArchiveRefused("unsupported_archive")


def key_and_field(name: str) -> tuple[str, str]:
    """WebDataset key and field: ``dir/sample001.seg.jpg`` -> (``dir/sample001``, ``seg.jpg``).

    The key ends at the basename's first dot; the field is everything after it (lowercase).
    """
    folder, base = posixpath.split(name)
    stem, _, ext = base.partition(".")
    return (f"{folder}/{stem}" if folder else stem), ext.lower()


def sample_to_row(sample: dict[str, Any]) -> dict[str, Any]:
    """A WebDataset sample as a table row (images stay bytes, text decoded).

    A json field's top-level keys become ``<field>.<key>`` columns (``json.question``); a field that does not decode
    is left out.
    """
    import json

    row: dict[str, Any] = {}
    for fld, data in sample.items():
        if fld == "__key__" or not isinstance(data, bytes):
            continue
        if fld.endswith(tuple(e.lstrip(".") for e in IMAGE_EXT)):
            row[fld] = data
        elif fld.endswith(("json", "jsonl")):
            try:
                value = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                row.update({f"{fld}.{k}": v for k, v in list(value.items())[:64] if isinstance(k, str)})
        else:
            try:
                row[fld] = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
    return row


def iter_samples(paths: list[Path], report: ShardReport) -> Iterator[dict[str, Any]]:
    """WebDataset samples ``{"__key__": key, <field>: bytes, ...}`` grouped by consecutive key, shard by shard."""
    for path in paths:
        current: dict[str, Any] | None = None
        for name, data in iter_shard(path, report):
            key, fld = key_and_field(name)
            if current is not None and current["__key__"] != key:
                yield current
                current = None
            current = current or {"__key__": key}
            current[fld] = data
        if current is not None:
            yield current


__all__ = [
    "ArchiveRefused",
    "ShardReport",
    "check_name",
    "iter_samples",
    "iter_shard",
    "key_and_field",
    "sample_to_row",
]
