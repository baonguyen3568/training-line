"""B-3b: a user's dataset folder is checked line by line, every problem reported at once, nothing guessed."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from vlm_pipeline.data.local_folder import scan_folder

Image = pytest.importorskip("PIL.Image")


def _img(path: Path, size: tuple[int, int] = (40, 30), fmt: str = "PNG") -> None:
    buf = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buf, format=fmt)
    path.write_bytes(buf.getvalue())


def _folder(tmp_path: Path, lines: list[object]) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "labels.jsonl").write_text("".join((json.dumps(x) if not isinstance(x, str) else x) + "\n" for x in lines))
    return d


def test_good_folder_gives_records_with_hashes(tmp_path: Path) -> None:
    d = _folder(
        tmp_path, [{"image": "a.png", "target": {"total": "1"}}, {"image": "sub/b.jpg", "target": "md", "prompt": "Q"}]
    )
    _img(d / "a.png")
    (d / "sub").mkdir()
    _img(d / "sub" / "b.jpg", fmt="JPEG")
    res = scan_folder(d)
    assert res.problem_count == 0
    assert [(r.image, r.prompt) for r in res.records] == [("a.png", None), ("sub/b.jpg", "Q")]
    assert all(len(r.sha256) == 64 for r in res.records)


def test_every_problem_is_reported_with_its_line(tmp_path: Path) -> None:
    outside = tmp_path / "secret.png"
    _img(outside)
    d = _folder(tmp_path, [
        "{not json",
        {"image": "../secret.png", "target": "x"},
        {"image": "/etc/passwd", "target": "x"},
        {"image": "gif.png", "target": "x"},
        {"image": "ok.png", "target": ""},
        {"image": "ok.png", "target": "x", "extra": 1},
        {"image": "link.png", "target": "x"},
        {"image": "big.png", "target": "x"},
        {"image": "ok.png", "target": "x"},
        {"image": "ok.png", "target": "x"},
        {"image": "nope.png", "target": "x"},
    ])  # fmt: skip
    _img(d / "gif.png", fmt="GIF")
    _img(d / "ok.png")
    _img(d / "big.png", size=(5000, 10))
    (d / "link.png").symlink_to(outside)
    res = scan_folder(d, max_image_side_px=4096)
    reasons = [(p["line"], p["reason"]) for p in res.problems]
    assert reasons == [
        (1, "not_json"), (2, "bad_image_name"), (3, "bad_image_name"), (4, "image_unreadable"), (5, "empty_target"),
        (6, "bad_keys"), (7, "bad_image_name"), (8, "image_too_big_side"), (10, "duplicate_image"),
        (11, "image_missing"),
    ]  # fmt: skip
    assert [r.image for r in res.records] == ["ok.png"]


def test_missing_labels_and_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert scan_folder(tmp_path / "empty").problems == [{"line": 0, "reason": "labels_missing"}]
    d = _folder(tmp_path, [])
    assert scan_folder(d).problems == [{"line": 0, "reason": "no_samples"}]


def test_max_samples_keeps_the_first_good_lines(tmp_path: Path) -> None:
    d = _folder(tmp_path, [{"image": f"{i}.png", "target": "x"} for i in range(5)])
    for i in range(5):
        _img(d / f"{i}.png")
    assert [r.image for r in scan_folder(d, max_samples=2).records] == ["0.png", "1.png"]


def test_a_symlink_inside_the_folder_is_not_a_file(tmp_path: Path) -> None:
    d = _folder(tmp_path, [{"image": "link.png", "target": "x"}])
    _img(d / "real.png")
    (d / "link.png").symlink_to(d / "real.png")  # stays inside, but a link is never followed
    assert [p["reason"] for p in scan_folder(d).problems] == ["image_not_a_file"]


@pytest.mark.parametrize("encoding", ["cp1258", "utf-16"])
def test_non_utf8_labels_are_line_problems_not_a_crash(tmp_path: Path, encoding: str) -> None:
    d = tmp_path / "data"
    d.mkdir()
    _img(d / "a.png")
    (d / "labels.jsonl").write_bytes('{"image": "a.png", "target": "Hóa đơn"}\n'.encode(encoding))
    res = scan_folder(d)
    assert res.records == []
    assert {p["reason"] for p in res.problems} == {"not_json"}
    assert res.problems[0]["line"] == 1


def test_names_are_normalized_or_refused_never_a_crash(tmp_path: Path) -> None:
    """ "./a.png" / "a//b.png" are how people write names (backend): stored normalized. Escapes stay refused (R1)."""
    names = ["./a.png", "sub//b.png", "a.png", "../a.png", "sub/../../a.png"]
    lines = [{"image": n, "target": "t"} for n in names]
    d = _folder(tmp_path, lines)
    _img(d / "a.png")
    (d / "sub").mkdir()
    _img(d / "sub" / "b.png")
    res = scan_folder(d)
    assert [r.image for r in res.records] == ["a.png", "sub/b.png"]
    assert [(p["line"], p["reason"]) for p in res.problems] == [
        (3, "duplicate_image"), (4, "bad_image_name"), (5, "bad_image_name")
    ]  # fmt: skip
