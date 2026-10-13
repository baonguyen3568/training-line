"""B7-3 step 1: offline vLLM batch, host side (budget under the 80 %-of-free rule, request checks, the job it runs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.common.runinfo import RunManifest, RunStatus
from vlm_pipeline.serve import vllm_runner
from vlm_pipeline.serve.vllm_batch import BatchRequest
from vlm_pipeline.serve.vllm_runner import BatchRequestError, build_items, engine_budget, run_vllm_batch

# ------------------------------------------------------------------------------------------------ budget


def test_budget_asks_mem_frac_when_it_fits() -> None:
    b = engine_budget(1.6, total_gb=121.7, available_gb=110.0)
    assert b.gpu_memory_utilization == pytest.approx(0.3)
    assert b.est_peak_gb == pytest.approx(0.3 * 121.7 + 4.0, abs=0.1)


def test_budget_shrinks_to_80_percent_of_free() -> None:
    b = engine_budget(1.6, total_gb=121.7, available_gb=30.0)  # 0.8 x 30 - 4 = 20 GB < 36.5 wanted
    assert b.engine_gb == pytest.approx(20.0)
    assert b.est_peak_gb <= 0.8 * 30.0 + 1e-6  # preflight passes: the job fits the user rule


def test_budget_never_starves_the_engine_it_queues_instead() -> None:
    b = engine_budget(1.6, total_gb=121.7, available_gb=5.0)  # allowed < 0: keep the floor, run_job waits
    assert b.engine_gb == pytest.approx(1.6 * 1.1 + 2.0, abs=0.1)
    assert b.est_peak_gb > 0.8 * 5.0


def test_model_bigger_than_mem_frac_is_a_caller_error() -> None:
    with pytest.raises(BatchRequestError, match="mem_frac"):
        engine_budget(40.0, total_gb=121.7, available_gb=110.0, mem_frac=0.3)


# ------------------------------------------------------------------------------------------------ items


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "data" / "imgs").mkdir(parents=True)
    (root / "data" / "imgs" / "a.png").write_bytes(b"x")
    model = root / ".cache" / "hf" / "hub" / "models--Q--M" / "snapshots" / "abc"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"\0" * 1024)
    return root


def test_items_keep_order_and_map_to_container_paths(tmp_path: Path) -> None:
    root = _project(tmp_path)
    items, dirs = build_items(root, [{"id": "b", "image": "data/imgs/a.png", "prompt": "p", "max_tokens": 8},
                                     {"id": "a", "prompt": "text only"}])  # fmt: skip
    assert [i.id for i in items] == ["b", "a"]
    assert items[0].image == "/workspace/data/imgs/a.png"
    assert items[1].image == ""
    assert dirs == [root / "data" / "imgs" / "a.png"]  # the file, not its dir: no sibling is exposed (security S2)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([{"id": "x"}, {"id": "x"}], "duplicated"),
        ([{"id": "x", "max_tokens": 0}], "max_tokens"),
        ([{"id": "x", "image": "data/imgs/missing.png"}], "regular file"),
        ([], "empty"),
    ],
)
def test_malformed_batches_are_refused(tmp_path: Path, raw: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(BatchRequestError, match=match):
        build_items(_project(tmp_path), raw)


def test_symlinked_or_outside_images_are_refused(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"x")
    (root / "data" / "imgs" / "link.png").symlink_to(outside)
    with pytest.raises(BatchRequestError, match="regular file"):
        build_items(root, [{"id": "x", "image": "data/imgs/link.png"}])
    with pytest.raises(BatchRequestError, match="outside"):
        build_items(root, [{"id": "x", "image": str(outside)}])


# ------------------------------------------------------------------------------------------------ the job


def test_job_is_narrowed_read_only_on_the_approved_vllm_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    model = root / ".cache" / "hf" / "hub" / "models--Q--M" / "snapshots" / "abc"
    seen: dict[str, Any] = {}

    def fake_run_job(job: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        seen.update(job=job, kw=kw, run_dir=run_dir)
        (run_dir / "out.jsonl").write_text(json.dumps({"id": "b", "text": "ok"}) + "\n")
        (run_dir / "engine.json").write_text(json.dumps({"vllm": "0.31.0"}))
        return manifest.model_copy(update={"status": RunStatus.COMPLETED, "container_digest": "sha256:bbe7"})

    monkeypatch.setattr(vllm_runner, "run_job", fake_run_job)
    res = run_vllm_batch(root, model, [{"id": "b", "image": "data/imgs/a.png", "prompt": "p"}], who="t",
                         min_pixels=200704, max_pixels=1003520, memory=(121.7, 110.0))  # fmt: skip
    job = seen["job"]
    assert (job.image, job.entrypoint) == ("vlm-vllm:0.31.0", "python3")
    assert job.argv[:2] == ("-m", "vlm_pipeline.serve.vllm_batch")
    assert job.narrow
    assert job.read_only_project
    assert job.rw_dirs == (res.run_dir,)
    assert set(job.ro_inputs) == {
        root / ".cache/hf/hub/models--Q--M",
        root / "data/imgs/a.png",
    }  # whole repo: blob links
    assert job.env["VLLM_NO_USAGE_STATS"] == "1"
    assert seen["kw"]["est_peak_gb"] == pytest.approx(0.3 * 121.7 + 4.0, abs=0.1)
    req = BatchRequest.from_json((res.run_dir / "batch.json").read_text())
    assert (req.min_pixels, req.max_pixels, req.enable_thinking) == (200704, 1003520, False)
    assert req.model_dir == "/workspace/.cache/hf/hub/models--Q--M/snapshots/abc"
    assert res.ok
    assert res.rows() == [{"id": "b", "text": "ok"}]
    assert res.engine()["container_digest"] == "sha256:bbe7"


def test_a_refused_batch_leaves_no_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    model = root / ".cache" / "hf" / "hub" / "models--Q--M" / "snapshots" / "abc"
    monkeypatch.setattr(vllm_runner, "run_job", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(BatchRequestError):
        run_vllm_batch(root, model, [{"id": "x"}, {"id": "x"}], who="t", memory=(121.7, 110.0))
    with pytest.raises(BatchRequestError, match="min_pixels"):
        run_vllm_batch(root, model, [{"id": "x"}], who="t", max_pixels=10, memory=(121.7, 110.0))
    assert not (root / "outputs").exists()


def test_a_big_batch_mounts_its_dir_but_uploads_stay_per_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    (root / "uploads").mkdir()
    monkeypatch.setattr(vllm_runner, "DIR_MOUNT_MIN", 3)
    raw = []
    for n in range(3):
        (root / "data" / "imgs" / f"{n}.png").write_bytes(b"x")
        (root / "uploads" / f"{n}.png").write_bytes(b"x")
        raw += [{"id": f"d{n}", "image": f"data/imgs/{n}.png"}, {"id": f"u{n}", "image": f"uploads/{n}.png"}]
    _, mounts = build_items(root, raw)
    assert mounts == [root / "data" / "imgs", *(root / "uploads" / f"{n}.png" for n in range(3))]


def test_a_model_without_safetensors_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    model = root / ".cache" / "hf" / "hub" / "models--Q--M" / "snapshots" / "abc"
    (model / "model.safetensors").unlink()
    (model / "pytorch_model.bin").write_bytes(b"pickle")
    monkeypatch.setattr(vllm_runner, "run_job", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(BatchRequestError, match="safetensors"):
        run_vllm_batch(root, model, [{"id": "x"}], who="t", memory=(121.7, 110.0))
