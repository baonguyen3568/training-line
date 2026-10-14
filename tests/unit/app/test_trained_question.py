"""H-prompt (P021): the question a model was taught reaches run.json, the model card, the trial and validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.local_records import DEFAULT_PROMPT, Record, single_prompt
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest


def _rec(prompt: str | None) -> Record:
    return Record(image="a.png", prompt=prompt, target="x", sha256="0" * 64)


def test_one_question_or_none() -> None:
    assert single_prompt([_rec("Đọc hoá đơn"), _rec("Đọc hoá đơn")], "chat") == "Đọc hoá đơn"
    assert single_prompt([_rec(None), _rec("")], "chat") == DEFAULT_PROMPT  # no question: the trainer's default
    assert single_prompt([_rec("a"), _rec("b")], "chat") is None
    assert single_prompt([_rec("a")], "nemotron_parse") == DEFAULT_PROMPT  # one fixed task
    assert single_prompt([], "chat") is None


def _run(root: Path, run_id: str, task: str, *, prompt: str | None = None, parent: str | None = None,
         stage: str = "train") -> Path:  # fmt: skip
    run_dir = root / "outputs" / "graph" / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest = RunManifest(run_id=run_id, name=run_id, stage=stage, task=task,  # type: ignore[arg-type]
                           status=RunStatus.COMPLETED, train_prompt=prompt, parent_run_id=parent)  # fmt: skip
    write_manifest(run_dir, manifest)
    return run_dir


def test_the_latest_training_decides(tmp_path: Path) -> None:
    from vlm_pipeline.export.notices import trained_prompt

    _run(tmp_path, "t1-a", "lora", prompt="Đọc hoá đơn")
    merged = _run(tmp_path, "g1-a", "merge_lora", parent="t1-a")
    (merged / "merged").mkdir()
    assert trained_prompt(merged / "merged", tmp_path) == "Đọc hoá đơn"
    _run(tmp_path, "t2-a", "sft", prompt=None, parent="g1-a")  # trained again on mixed questions
    assert trained_prompt(tmp_path / "outputs" / "graph" / "runs" / "t2-a", tmp_path) is None  # never the older one
    assert trained_prompt(tmp_path / "nowhere", tmp_path) is None


def test_the_model_card_says_how_to_ask_in_a_block_nothing_closes() -> None:
    from vlm_pipeline.export.notices import Modification, _model_card

    tricky = "Read ```` then # heading"
    card = _model_card([], [Modification(run_id="t", task="lora", dataset=None, ended_at=None, train_prompt=tricky)],
                       None)  # fmt: skip
    assert "## How to ask" in card
    assert "`````text\n" + tricky + "\n`````" in card  # a fence longer than any backtick run inside
    assert "## How to ask" not in _model_card([], [Modification(run_id="t", task="lora", dataset=None,
                                                                ended_at=None)], None)  # fmt: skip


@pytest.mark.parametrize(("box", "fixed", "trained", "seen"), [
    ("Tổng tiền?", False, "Đọc hoá đơn", ("Tổng tiền?", "user")),
    ("", False, "Đọc hoá đơn", ("Đọc hoá đơn", "trained")),
    ("", False, None, (DEFAULT_PROMPT, "default")),
    ("x", True, "Đọc hoá đơn", ("", "fixed")),
])  # fmt: skip
def test_an_empty_box_asks_the_trained_question(monkeypatch: pytest.MonkeyPatch, box: str, fixed: bool,
                                                trained: str | None, seen: tuple[str, str]) -> None:  # fmt: skip
    from types import SimpleNamespace

    from vlm_pipeline.app import trial_worker
    from vlm_pipeline.export import notices
    from vlm_pipeline.nodes import REGISTRY

    spec = REGISTRY.get("serve.vllm")
    assert spec is not None
    monkeypatch.setattr(notices, "trained_prompt", lambda artifact, root: trained)
    worker = SimpleNamespace(root=Path("/r"), host_path=Path)
    job = SimpleNamespace(params=spec.validate_params({"image": "a" * 32, "prompt": box}),
                          inputs={"model": SimpleNamespace(path="/r/m")})  # fmt: skip
    entry = SimpleNamespace(task_prompt_fixed=fixed)
    assert trial_worker.question_for(worker, job, entry) == seen  # type: ignore[arg-type]


def _n(i: str, t: str, **p: Any) -> dict[str, Any]:
    return {"id": i, "type": t, "version": "1.0.0", "params": p}


def _e(a: str, out: str, b: str, port: str) -> dict[str, list[str]]:
    return {"from": [a, out], "to": [b, port]}


def _wf(data: dict[str, Any]) -> dict[str, Any]:
    return {"format": "vlm-workflow", "schema_version": 1, "name": "t",
            "nodes": [_n("m", "model.hf_pick"), data, _n("s", "data.split"), _n("t", "train.lora_vlm"),
                      _n("g", "train.merge_lora"), _n("v", "serve.vllm", image="a" * 32)],
            "edges": [_e("d", "dataset", "s", "dataset"), _e("s", "train", "t", "train_data"),
                      _e("m", "model", "t", "base_model"), _e("m", "model", "g", "base_model"),
                      _e("t", "adapter", "g", "adapter"), _e("g", "model", "v", "model")]}  # fmt: skip


def _hints(root: Path, wf: dict[str, Any]) -> list[tuple[Any, ...]]:
    from vlm_pipeline.graph import validate
    from vlm_pipeline.nodes import REGISTRY

    issues = validate(wf, REGISTRY, workspace=root)
    return [(i.code, i.node_id, i.param, i.field_hint, i.params["prompt"]) for i in issues if i.code == "MODEL-007"]


def _folder(root: Path, prompts: list[str | None]) -> None:
    from PIL import Image

    folder = root / "data" / "inv"
    folder.mkdir(parents=True)
    rows = []
    for n, p in enumerate(prompts):
        Image.new("RGB", (8, 8)).save(folder / f"{n}.png")
        rows.append({"image": f"{n}.png", "target": "x", **({"prompt": p} if p else {})})
    (folder / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


@pytest.mark.parametrize(("prompts", "hint"), [
    (["Đọc hoá đơn", "Đọc hoá đơn"], "Đọc hoá đơn"),
    ([None, None], DEFAULT_PROMPT),
    (["a", "b"], None),
])  # fmt: skip
def test_validate_hints_the_taught_question_of_a_local_folder(tmp_path: Path, prompts: list[str | None],
                                                              hint: str | None) -> None:  # fmt: skip
    _folder(tmp_path, prompts)
    got = _hints(tmp_path, _wf(_n("d", "data.local_folder", path="data/inv")))
    assert got == ([("MODEL-007", "v", "prompt", True, hint)] if hint else [])


def test_a_user_dataset_hints_its_shared_question_unless_rows_bring_their_own(tmp_path: Path) -> None:
    from vlm_pipeline.common.catalog import set_user_dataset_lookup, user_dataset_id

    ds = user_dataset_id("org/d", "c" * 40, "default", "train")
    status: dict[str, object] = {"name": "d", "repo": "org/d", "revision": "c" * 40, "rows": 2, "download": None,
                                 "license": "mit", "prompt": "Đọc hoá đơn"}  # fmt: skip
    known = {ds: status}
    set_user_dataset_lookup(known.get)
    try:
        wf = _wf(_n("d", "data.hf_load", user_dataset=ds, user_repo="org/d", user_revision="c" * 40))
        assert _hints(tmp_path, wf) == [("MODEL-007", "v", "prompt", True, "Đọc hoá đơn")]
        known[ds]["prompt"] = None
        assert _hints(tmp_path, wf) == []
    finally:
        set_user_dataset_lookup(None)


def test_no_hint_outside_the_project_or_for_an_untrained_model(tmp_path: Path) -> None:
    assert _hints(tmp_path, _wf(_n("d", "data.local_folder", path="../../etc"))) == []
    wf = {"format": "vlm-workflow", "schema_version": 1, "name": "t",
          "nodes": [_n("m", "model.hf_pick"), _n("v", "serve.vllm", image="a" * 32)],
          "edges": [_e("m", "model", "v", "model")]}  # fmt: skip
    assert _hints(tmp_path, wf) == []


def test_a_question_too_long_to_record_is_none() -> None:
    """security: a question over the cap is never cut (a cut question is not the one trained), and never fails a run."""
    long = "x" * 2001
    assert single_prompt([_rec(long), _rec(long)], "chat") is None
