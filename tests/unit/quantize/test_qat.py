"""quantize.qat (ADR-022 #10): host-side pieces of the container worker (data file, step count, LR schedule)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_pipeline.common.job_kinds import argv_allowed, job_entrypoint
from vlm_pipeline.quantize.qat import GRAD_ACCUM, WARMUP_STEPS, lr_at, optimizer_steps, read_samples


def test_steps_follow_epochs_and_the_cap() -> None:
    assert optimizer_steps(100, 1, 0) == 13  # ceil(100 / 8)
    assert optimizer_steps(100, 3, 0) == 39
    assert optimizer_steps(100, 3, 20) == 20
    assert optimizer_steps(GRAD_ACCUM, 1, 0) == 1


def test_lr_warms_up_then_decays_to_a_tenth() -> None:
    peak, total = 1e-4, 40
    assert lr_at(0, total, peak) == pytest.approx(peak / WARMUP_STEPS)
    assert lr_at(WARMUP_STEPS - 1, total, peak) == pytest.approx(peak)
    assert lr_at(total, total, peak) == pytest.approx(0.1 * peak)
    lrs = [lr_at(s, total, peak) for s in range(WARMUP_STEPS, total)]
    assert lrs == sorted(lrs, reverse=True)


def test_samples_need_exactly_image_prompt_answer(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    good = {"image": "a.png", "prompt": "Read.", "answer": "{}"}
    f = tmp_path / "qat_data.jsonl"
    f.write_text(json.dumps(good) + "\n\n")
    assert read_samples(f, images) == [{**good, "image": str(images.resolve() / "a.png")}]
    for bad in ({**good, "extra": 1}, {**good, "answer": ""}, {**good, "prompt": 3}):
        f.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match=r"qat_data\.jsonl:1"):
            read_samples(f, images)
    f.write_text("")
    with pytest.raises(ValueError, match="no samples"):
        read_samples(f, images)


@pytest.mark.parametrize("name", ["/etc/passwd", "../x.png", "a/../../x.png", "a\\b.png", "a\x00.png"])
def test_sample_images_stay_inside_the_images_folder(tmp_path: Path, name: str) -> None:
    f = tmp_path / "qat_data.jsonl"
    f.write_text(json.dumps({"image": name, "prompt": "p", "answer": "a"}) + "\n")
    with pytest.raises(ValueError, match="inside the images folder"):
        read_samples(f, tmp_path)


def test_a_link_inside_the_images_folder_cannot_point_out(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    (images / "out.png").symlink_to(tmp_path / "secret.png")
    f = tmp_path / "qat_data.jsonl"
    f.write_text(json.dumps({"image": "out.png", "prompt": "p", "answer": "a"}) + "\n")
    with pytest.raises(ValueError, match="inside the images folder"):
        read_samples(f, images)


def test_qat_runs_through_the_ngc_wrapper_as_a_quantize_job() -> None:
    argv = ("python", "-m", "vlm_pipeline.quantize.qat", "--model", "/m")
    assert argv_allowed("quantize", argv)
    assert not argv_allowed("train", argv)
    assert job_entrypoint("quantize", argv) == ("/opt/nvidia/nvidia_entrypoint.sh", argv)
