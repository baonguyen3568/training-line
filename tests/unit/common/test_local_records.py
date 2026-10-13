"""Local dataset records (B-3b): target text per model family, stable split, make_meta_dataset files."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_pipeline.common import local_records as lr

SHA = "0" * 64


def _r(name: str, target: object, prompt: str | None = None) -> lr.Record:
    return lr.Record(image=name, target=target, prompt=prompt, sha256=SHA)


def test_json2token_matches_automodel_cord_format() -> None:
    gt = {
        "menu": [{"nm": "Cơm", "price": "35,000"}, {"nm": "Trà", "price": "5,000"}],
        "total": {"total_price": "40,000"},
    }
    assert lr.json2token(gt) == (
        "<s_total><s_total_price>40,000</s_total_price></s_total>"
        "<s_menu><s_price>35,000</s_price><s_nm>Cơm</s_nm><sep/><s_price>5,000</s_price><s_nm>Trà</s_nm></s_menu>"
    )
    assert lr.json2token({"text_sequence": "raw"}) == "raw"


def test_targets_per_family_and_prompts() -> None:
    parse = next(iter(lr.conversations_for([_r("a.png", {"k": "v"}, prompt="ignored")], "nemotron_parse")))
    assert parse["messages"][0]["content"] == f"<image>\n{lr.DEFAULT_PROMPT}"  # Parse: fixed task, prompt ignored
    assert parse["messages"][1]["content"] == "<s_k>v</s_k>"
    chat = next(iter(lr.conversations_for([_r("a.png", {"k": "v"}, prompt="Read it")], "chat")))
    assert chat["messages"][0]["content"] == "<image>\nRead it"
    assert json.loads(chat["messages"][1]["content"]) == {"k": "v"}
    assert chat["images"] == ["a.png"]
    plain = next(iter(lr.conversations_for([_r("b.png", "markdown text")], "chat")))
    assert plain["messages"][0]["content"] == f"<image>\n{lr.DEFAULT_PROMPT}"
    assert plain["messages"][1]["content"] == "markdown text"


def test_split_is_stable_disjoint_and_close_to_the_fraction() -> None:
    records = [_r(f"img_{i:04d}.png", "t") for i in range(2000)]
    train = lr.select(records, part="train", seed=7, val_fraction=0.1)
    val = lr.select(records, part="validation", seed=7, val_fraction=0.1)
    assert {r.image for r in train}.isdisjoint({r.image for r in val})
    assert len(train) + len(val) == 2000
    assert 150 < len(val) < 250
    assert lr.select(records[::-1], part="validation", seed=7, val_fraction=0.1)[::-1] == val  # order-independent
    assert lr.select(records, part="validation", seed=8, val_fraction=0.1) != val  # the seed matters
    assert len(lr.select(records, part="train", seed=7, val_fraction=0.1, limit=10)) == 10


def test_meta_dataset_files(tmp_path: Path) -> None:
    meta = lr.write_meta_dataset(tmp_path, "train", [_r("a.png", "x")], "chat", media_dir="/workspace/data/mine")
    cfg = json.loads(meta.read_text())
    assert cfg == {"train": {"file_name": "train.jsonl", "media_dir": "/workspace/data/mine"}}
    assert json.loads((tmp_path / "train.jsonl").read_text())["images"] == ["a.png"]
    sha = lr.write_records(tmp_path / "records.jsonl", [_r("a.png", {"k": 1})])
    assert len(sha) == 64
    assert list(lr.read_records(tmp_path / "records.jsonl")) == [_r("a.png", {"k": 1})]
