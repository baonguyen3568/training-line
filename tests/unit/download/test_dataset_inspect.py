"""ADR-021 / P021 H4: a Hugging Face dataset's layout, its resolve, and the network-none inspect job."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlm_pipeline.download import hub
from vlm_pipeline.download import inspect as ins
from vlm_pipeline.download.dataset_layout import choose, classify, has_script, layout, sample_files

NAMES = ["README.md", "data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
         "data/validation-00000-of-00001.parquet", "DocVQA/test-00000.parquet", "DocVQA/validation-00000.parquet",
         "imgs/train/metadata.jsonl", "imgs/train/a.png", "loader.py"]  # fmt: skip


# ------------------------------------------------------------------------------------------- layout
def test_configs_and_splits_follow_the_hub_layouts() -> None:
    tree = layout(NAMES)
    assert sorted(tree) == ["DocVQA", "default", "imgs"]
    assert sorted(tree["default"]) == ["train", "validation"]
    assert sorted(tree["DocVQA"]) == ["test", "validation"]
    assert tree["imgs"]["train"][0].format == "imagefolder"
    assert classify("README.md") is None
    assert classify("loader.py") is None
    assert has_script(NAMES)


@pytest.mark.parametrize(("config", "split", "chosen"), [
    (None, None, ("default", "train")), ("DocVQA", None, ("DocVQA", "test")), ("DocVQA", "validation",
    ("DocVQA", "validation")), ("nope", "nope", ("default", "train"))])  # fmt: skip
def test_the_choice_falls_back_to_default_and_train(config: str | None, split: str | None,
                                                    chosen: tuple[str, str]) -> None:  # fmt: skip
    assert choose(layout(NAMES), config, split) == chosen


def test_the_sample_is_the_smallest_file_and_never_over_the_cap() -> None:
    files = layout(NAMES)["default"]["train"]
    sizes = {"data/train-00000-of-00002.parquet": 50, "data/train-00001-of-00002.parquet": 10}
    assert sample_files(files, sizes, NAMES) == ("parquet", ["data/train-00001-of-00002.parquet"])
    huge = dict.fromkeys(sizes, 1 << 40)
    assert sample_files(files, huge, NAMES) == ("parquet", [])  # too big to sample: said, never guessed


# ------------------------------------------------------------------------------------------- resolve (job)
class _DatasetHub:
    def __init__(self, siblings: list[Any], contents: dict[str, bytes], *, export: bool = False) -> None:
        self.siblings, self.contents, self.export, self.fetched = siblings, contents, export, []

    def dataset_info(self, repo: str, revision: str, files_metadata: bool) -> Any:
        return SimpleNamespace(siblings=self.siblings, sha="d" * 40, gated=False, card_data={"license": "mit"})

    def list_repo_refs(self, repo: str, repo_type: str) -> Any:
        converts = [SimpleNamespace(ref="refs/convert/parquet")] if self.export else []
        return SimpleNamespace(converts=converts)

    def fetch(self, *, filename: str, local_dir: str, **_: Any) -> str:
        self.fetched.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.contents.get(filename, b""))
        return str(path)


def _sib(name: str, size: int) -> Any:
    return SimpleNamespace(rfilename=name, size=size, lfs=None, blob_id=None)


def _run_dir(tmp_path: Path, **spec: Any) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = {"kind": "dataset", "repo": "org/d", "revision": "main", "max_bytes": 10_000, "resolve": True, **spec}
    (run_dir / hub.SPEC_NAME).write_text(json.dumps(body))
    return run_dir


def test_a_dataset_resolve_lists_configs_and_fetches_one_sample(tmp_path: Path) -> None:
    sibs = [_sib("data/train-0.parquet", 30), _sib("data/train-1.parquet", 20), _sib("data/test-0.parquet", 5),
            _sib("script.py", 3)]  # fmt: skip
    fake = _DatasetHub(sibs, {}, export=True)
    run_dir = _run_dir(tmp_path)
    assert hub.run(run_dir, api=fake, fetch=fake.fetch) == 0
    meta = json.loads((run_dir / "stage" / "resolve.json").read_text())["dataset"]
    assert (meta["config"], meta["split"], meta["splits"], meta["format"]) == ("default", "train", ["test", "train"],
                                                                                "parquet")  # fmt: skip
    assert (meta["has_script"], meta["has_parquet_export"], meta["split_bytes"]) == (True, True, 50)
    assert fake.fetched == ["data/train-1.parquet"]  # the smallest file of the split, nothing else
    assert "script.py" not in fake.fetched


@pytest.mark.parametrize("bad", [{"config": "../x"}, {"split": "a b"}, {"config": 3}])
def test_a_dataset_resolve_refuses_strange_names(tmp_path: Path, bad: dict[str, Any]) -> None:
    fake = _DatasetHub([], {})
    assert hub.run(_run_dir(tmp_path, **bad), api=fake, fetch=fake.fetch) == hub.EXIT_BAD_SPEC


# ------------------------------------------------------------------------------------------- inspect (data job)
def _png(color: tuple[int, int, int] = (200, 10, 10), size: tuple[int, int] = (600, 300)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_jsonl_with_question_and_answer_is_mapped(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    rows = [{"question": "Q\x00 one?\x1b[31m", "answer": "A1", "id": 1}, {"question": "Q2", "answer": "A2", "id": 2}]
    (sample / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out"
    assert ins.main(["--sample", str(sample), "--format", "jsonl", "--out", str(out)]) == 0
    result = json.loads((out / "inspect.json").read_text())
    assert result["suggested"] == {"question": "question", "answer": "answer"}
    assert {c["name"]: c["kind"] for c in result["columns"]} == {"question": "text", "answer": "text", "id": "number"}
    assert result["samples"][0] == {"question": "Q one?[31m", "answer": "A1"}  # control characters dropped


def test_messages_win_over_question_answer(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], "question": "q"}
    (sample / "train.jsonl").write_text(json.dumps(row))
    result = ins.inspect(sample, "jsonl", tmp_path / "out")
    assert result["suggested"] == {"messages": "messages"}
    assert result["samples"][0] == {"question": "hi", "answer": "yo"}


def test_imagefolder_thumbnails_are_small_pngs_and_stay_inside(tmp_path: Path) -> None:
    sample = tmp_path / "sample" / "train"
    sample.mkdir(parents=True)
    (sample / "a.png").write_bytes(_png())
    (tmp_path / "secret.png").write_bytes(_png())
    (sample / "link.png").symlink_to(sample / "a.png")
    meta = [{"file_name": "a.png", "text": "red"}, {"file_name": "../../secret.png", "text": "x"},
            {"file_name": "missing.png", "text": "y"}, {"file_name": "link.png", "text": "z"}]  # fmt: skip
    (sample / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in meta))
    out = tmp_path / "out"
    result = ins.inspect(tmp_path / "sample", "imagefolder", out)
    assert result["suggested"]["image"] == "file_name"
    assert result["samples"][0]["thumb"] == 0
    errors = [s.get("error") for s in result["samples"][1:]]
    assert errors == ["image_missing", "image_missing", "image_missing"]  # never outside, never through a link
    from PIL import Image

    with Image.open(out / "thumbs" / "0.png") as thumb:
        assert max(thumb.size) <= ins.THUMB_SIDE
        assert thumb.format == "PNG"


def test_image_bytes_cells_decode_under_the_rules(tmp_path: Path) -> None:
    rows = [{"image": {"bytes": _png(), "path": None}, "caption": "c"},
            {"image": {"bytes": b"\x89PNGnot", "path": None}, "caption": "d"}]  # fmt: skip
    cols = ins.columns(rows)
    mapping = ins.suggest(cols)
    assert mapping == {"image": "image", "answer": "caption"}
    samples = ins.preview(rows, cols, mapping, tmp_path, tmp_path / "thumbs")
    assert samples[0]["thumb"] == 0
    assert samples[1]["error"] == "image_unreadable"


@pytest.mark.parametrize("content", [b"{not json", b'{"a": 1}\n' + b"x" * (ins.MAX_LINE_BYTES + 1)])
def test_an_unreadable_sample_is_reported_not_guessed(tmp_path: Path, content: bytes) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "train.jsonl").write_bytes(content)
    out = tmp_path / "out"
    assert ins.main(["--sample", str(sample), "--format", "jsonl", "--out", str(out)]) == 0
    assert json.loads((out / "inspect.json").read_text())["error"] == "unreadable"


def test_a_symlinked_sample_is_never_read(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    (tmp_path / "outside.jsonl").write_text(json.dumps({"question": "q", "answer": "a"}))
    (sample / "train.jsonl").symlink_to(tmp_path / "outside.jsonl")
    assert ins.inspect(sample, "jsonl", tmp_path / "out")["error"] == "no_sample"


def test_csv_columns_and_long_text_is_truncated(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "train.csv").write_text("prompt,response\n" + "p," + "r" * 2000 + "\n")
    result = ins.inspect(sample, "csv", tmp_path / "out")
    assert result["suggested"] == {"question": "prompt", "answer": "response"}
    assert len(result["samples"][0]["answer"]) == ins.MAX_TEXT
