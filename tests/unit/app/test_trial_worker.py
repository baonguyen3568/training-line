"""B-7 "Chạy thử" host side: image checks before the GPU lock, what the container loads, and how results come back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vlm_pipeline.app import trial_worker
from vlm_pipeline.app.trial_worker import TrialImageError, build_request, check_image, run_trial
from vlm_pipeline.app.workers import RealWorker
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest
from vlm_pipeline.graph.errors import ContainerExit
from vlm_pipeline.graph.worker import ArtifactRef, NodeJob
from vlm_pipeline.nodes import REGISTRY

UID = "0123456789abcdef0123456789abcdef"
OTHER = "f" * 32


def _root(tmp_path: Path) -> Path:
    from vlm_pipeline.common.paths import project_root

    root = tmp_path / "proj"
    (root / "configs" / "train").mkdir(parents=True)
    (root / "configs" / "train" / "templates").symlink_to(project_root() / "configs" / "train" / "templates")
    (root / "uploads").mkdir()
    (root / "uploads" / f"{UID}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return root


def _job(params: dict[str, Any], run_dir: Path, model: ArtifactRef) -> NodeJob:
    spec = REGISTRY.get("serve.vllm")
    assert spec is not None
    return NodeJob(
        node_id="try", spec=spec, params=spec.validate_params(params), inputs={"model": model}, run_dir=run_dir
    )


def _snap(root: Path, repo: str, rev: str) -> Path:
    d = root / ".cache" / "hf" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def _base_ref(w: RealWorker, model_id: str) -> tuple[Any, ArtifactRef]:
    entry = w.catalog.get(model_id)
    assert entry is not None
    path = _snap(w.root, entry.repo, entry.revision)
    return entry, ArtifactRef("Model", str(path.resolve()), f"{entry.repo}@{entry.revision}")


@pytest.mark.parametrize(("raw", "reason"), [("", "missing"), (OTHER, "missing"), ("dir", "not_a_file")])
def test_image_checks_before_the_gpu_lock(tmp_path: Path, raw: str, reason: str) -> None:
    root = _root(tmp_path)
    (root / "uploads" / "dir.png").mkdir()  # exists, but not a regular file
    with pytest.raises(TrialImageError) as ei:
        check_image(root, "try", raw)
    assert ei.value.code == "DATA-005"
    assert ei.value.params == {"node": "try", "reason": reason}


def test_params_take_an_upload_id_never_a_path() -> None:
    spec = REGISTRY.get("serve.vllm")
    assert spec is not None
    for bad in ("uploads/a.png", "../x", "/etc/passwd", UID.upper()):
        with pytest.raises(ValueError, match="pattern"):
            spec.validate_params({"image": bad})


def test_symlinked_and_oversized_uploads_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(tmp_path)
    (tmp_path / "secret.png").write_bytes(b"x")
    (root / "uploads" / f"{OTHER}.png").symlink_to(tmp_path / "secret.png")
    with pytest.raises(TrialImageError, match="not an uploaded file"):
        check_image(root, "try", OTHER)
    monkeypatch.setattr(trial_worker, "MAX_IMAGE_BYTES", 8)
    with pytest.raises(TrialImageError) as ei:
        check_image(root, "try", UID)
    assert ei.value.params["reason"] == "too_large"
    monkeypatch.undo()
    assert check_image(root, "try", UID) == (root / "uploads" / f"{UID}.png").resolve()


def test_base_chat_model_uses_its_training_pixels_and_the_prompt(tmp_path: Path) -> None:
    w = RealWorker(_root(tmp_path))
    entry, ref = _base_ref(w, "cosmos-reason2-2b")
    job = _job({"image": UID, "prompt": "Tổng tiền?"}, w.root / "run", ref)
    req, got = build_request(w, job, w.root / "uploads" / f"{UID}.png")
    assert got.id == entry.id
    assert req.family == "chat"
    assert (req.min_pixels, req.max_pixels) == (200704, 1003520)  # the cosmos training template's processor
    assert req.prompt == "Tổng tiền?"
    assert req.model_dir == req.processor_dir == req.generation_config_dir
    assert req.adapter_dir is None
    assert req.image == f"/workspace/uploads/{UID}.png"


def test_lora_output_is_loaded_on_its_base_with_the_run_pixels(tmp_path: Path) -> None:
    w = RealWorker(_root(tmp_path))
    entry, base = _base_ref(w, "cosmos-reason2-2b")
    run_dir = w.root / "outputs" / "graph" / "runs" / "ft-1"
    adapter = run_dir / "checkpoints" / "epoch_0_step_9" / "model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (run_dir / "config.yaml").write_text("processor:\n  min_pixels: 100\n  max_pixels: 400\n")
    write_manifest(
        run_dir,
        RunManifest(
            run_id="ft-1",
            name="ft",
            stage="train",
            task="lora",
            status=RunStatus.COMPLETED,
            base_model=entry.repo,
            base_model_revision=entry.revision,
        ),
    )
    job = _job({"image": UID}, w.root / "run", ArtifactRef("Model", str(adapter), "ft-1:abc"))
    req, _ = build_request(w, job, w.root / "uploads" / f"{UID}.png")
    assert req.model_dir == w.container_path(Path(base.path))
    assert req.adapter_dir == w.container_path(adapter)
    assert (req.min_pixels, req.max_pixels) == (100, 400)
    assert req.prompt == "Describe this image."


def test_parse_ignores_the_prompt_and_reads_the_prepared_base(tmp_path: Path) -> None:
    w = RealWorker(_root(tmp_path))
    entry = w.catalog.get("nemotron-parse-v1.1")
    assert entry is not None
    assert entry.task_prompt_fixed
    prepared = w.root / ".cache" / "prepared" / "nemotron-parse-v1.1_fd1e3c6b" / "model"
    prepared.mkdir(parents=True)
    (prepared / "config.json").write_text("{}")
    ref = ArtifactRef("Model", str(prepared), f"{entry.repo}@{entry.revision}")
    job = _job({"image": UID, "prompt": "ignored"}, w.root / "run", ref)
    req, _ = build_request(w, job, w.root / "uploads" / f"{UID}.png")
    assert req.family == "nemotron_parse"
    assert req.prompt == ""
    assert req.decode_overrides == {"repetition_penalty": 1.0}  # template 001a §10.7 over the base 1.1, as in eval


def _fake_job(result: dict[str, Any] | None, status: RunStatus = RunStatus.COMPLETED) -> Any:
    def fake_run_job(container: Any, manifest: RunManifest, run_dir: Path, **kw: Any) -> RunManifest:
        assert container.argv[:3] == ("python", "-m", "vlm_pipeline.serve.trial")
        assert kw["require_approved_image"] is True
        assert container.read_only_project
        assert container.rw_dirs == (run_dir,)
        assert json.loads((run_dir / "request.json").read_text())["family"] == "chat"
        if result is not None:
            (run_dir / "result.json").write_text(json.dumps(result))
        return manifest.model_copy(update={"status": status, "exit_code": 0 if status == RunStatus.COMPLETED else 1})

    return fake_run_job


def test_answer_comes_back_as_the_endpoint_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    _, ref = _base_ref(w, "cosmos-reason2-2b")
    run_dir = w.root / "outputs" / "graph" / "runs" / "try-1"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text('{"text": "stale"}')
    monkeypatch.setattr(trial_worker, "run_job", _fake_job({"text": "42.000đ", "engine_used": "hf"}))
    out = run_trial(w, _job({"image": UID, "engine": "vllm"}, run_dir, ref)).outputs["endpoint"]
    assert out.type == "Endpoint"
    assert out.hash.startswith("sha256:")  # content identity of result.json, not the run id
    result = json.loads(Path(out.path).read_text())
    assert result["text"] == "42.000đ"
    assert (result["engine_used"], result["engine_requested"]) == ("hf", "vllm")  # never silently "vllm"


def test_container_side_image_rejection_is_a_friendly_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    _, ref = _base_ref(w, "cosmos-reason2-2b")
    run_dir = w.root / "run"
    error = {"error": {"code": "TRIAL_IMAGE", "reason": "too_many_pixels"}}
    monkeypatch.setattr(trial_worker, "run_job", _fake_job(error, RunStatus.FAILED))
    with pytest.raises(TrialImageError) as ei:
        run_trial(w, _job({"image": UID}, run_dir, ref))
    assert ei.value.params == {"node": "try", "reason": "too_many_pixels"}


def test_failed_job_without_answer_is_a_container_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = RealWorker(_root(tmp_path))
    _, ref = _base_ref(w, "cosmos-reason2-2b")
    monkeypatch.setattr(trial_worker, "run_job", _fake_job(None, RunStatus.FAILED))
    with pytest.raises(ContainerExit):
        run_trial(w, _job({"image": UID}, w.root / "run", ref))


def test_temperature_turns_on_sampling_with_top_p() -> None:
    from vlm_pipeline.serve.trial import TrialRequest, decoding_overrides

    req = TrialRequest("m", "g", "p", "chat", "i.png", "q", max_new_tokens=256, temperature=0.0, top_p=0.9)
    assert decoding_overrides(req) == {"max_new_tokens": 256}  # greedy: the base config's sampling stays off
    hot = TrialRequest("m", "g", "p", "chat", "i.png", "q", max_new_tokens=256, temperature=0.7, top_p=0.9)
    assert decoding_overrides(hot) == {"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_p": 0.9}


def test_model_without_a_catalog_base_is_refused(tmp_path: Path) -> None:
    from vlm_pipeline.app.workers import UnknownModelError

    w = RealWorker(_root(tmp_path))
    stray = w.root / "outputs" / "elsewhere" / "model"
    stray.mkdir(parents=True)
    (tmp_path / "run.json").write_text("{}")  # outside the project: must never be read
    job = _job({"image": UID}, w.root / "run", ArtifactRef("Model", str(stray), "someone:else"))
    with pytest.raises(UnknownModelError):
        build_request(w, job, w.root / "uploads" / f"{UID}.png")
