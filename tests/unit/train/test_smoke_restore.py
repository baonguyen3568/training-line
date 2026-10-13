"""``smoke --restore-from``: the resume checkpoint is injected into the checkpoint block of the recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.train import smoke

RECIPE = "model:\n  x: 1\ncheckpoint:\n  enabled: true\n  checkpoint_dir: /w/run/checkpoints\nloss_fn:\n  y: 2\n"


def test_restore_from_goes_into_checkpoint_block() -> None:
    out = smoke._with_restore_from(RECIPE, "/w/outputs/runs/a/checkpoints/epoch_0_step_1")
    assert "checkpoint:\n  restore_from: /w/outputs/runs/a/checkpoints/epoch_0_step_1\n  enabled: true\n" in out
    assert out.replace("  restore_from: /w/outputs/runs/a/checkpoints/epoch_0_step_1\n", "") == RECIPE


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("model:\n  x: 1\n", "no top-level checkpoint"),
        ("model:\n  checkpoint:\n    a: 1\n", "no top-level checkpoint"),
        ("checkpoint:\n  restore_from: LATEST\n", "already sets"),
    ],
)
def test_restore_from_refuses_ambiguous_configs(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        smoke._with_restore_from(text, "/w/c")


FUSED = "optimizer:\n  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam\n  lr: 1.0e-5\n"
ADAMW = "optimizer:\n  lr: 1.0e-5\n  _target_: torch.optim.AdamW\n"


def _ckpt(tmp_path: Path, parent_cfg: str | None, parts: tuple[str, ...] = smoke._CKPT_PARTS) -> Path:
    ckpt = tmp_path / "epoch_0_step_1"
    ckpt.mkdir()
    for part in parts:
        (ckpt / part).mkdir()
    if parent_cfg is not None:
        (ckpt / "config.yaml").write_text("model:\n  x: 1\n" + parent_cfg + "loss_fn:\n  y: 2\n", encoding="utf-8")
    return ckpt


def test_restore_source_accepts_same_optimizer(tmp_path: Path) -> None:
    assert smoke._check_restore_source(_ckpt(tmp_path, FUSED), RECIPE + FUSED) is None


def test_restore_source_refuses_other_optimizer(tmp_path: Path) -> None:
    problem = smoke._check_restore_source(_ckpt(tmp_path, ADAMW), RECIPE + FUSED)
    assert problem is not None
    assert "torch.optim.AdamW" in problem


def test_restore_source_refuses_other_state_layout(tmp_path: Path) -> None:
    parent = FUSED + "  master_weights: false  # no fp32 master\n"
    child = FUSED.replace("lr: 1.0e-5", "lr: 2.0e-5") + "  master_weights: true\n"
    problem = smoke._check_restore_source(_ckpt(tmp_path, parent), RECIPE + child)
    assert problem == "optimizer differs from the checkpoint's: master_weights False -> True"


def test_restore_source_reads_automodel_redump(tmp_path: Path) -> None:
    """The checkpoint's config.yaml is AutoModel's re-dump of the recipe: same values, other spelling."""
    recipe = FUSED + "  betas: [0.9, 0.95]\n"
    redump = FUSED.replace("1.0e-5", "1.0e-05") + "  betas:\n  - 0.9\n  - 0.95\n"
    recipe += "  eps: 1e-8\n"  # YAML 1.1: a string; the re-dump writes the float 1.0e-08
    redump += "  eps: 1.0e-08\n"
    assert smoke._check_restore_source(_ckpt(tmp_path, redump), RECIPE + recipe) is None


def test_restore_source_allows_lr_and_weight_decay_changes(tmp_path: Path) -> None:
    child = FUSED.replace("lr: 1.0e-5", "lr: 2.0e-5  # resume with a higher lr") + "  weight_decay: 0.1\n"
    assert smoke._check_restore_source(_ckpt(tmp_path, FUSED), RECIPE + child) is None


@pytest.mark.parametrize(
    ("parent", "child", "match"),
    [
        (None, FUSED, "no config.yaml"),
        ("", FUSED, "only in the recipe"),
        (FUSED, "", "only in the checkpoint"),
    ],
)
def test_restore_source_refuses_unverifiable(tmp_path: Path, parent: str | None, child: str, match: str) -> None:
    problem = smoke._check_restore_source(_ckpt(tmp_path, parent), RECIPE + child)
    assert problem is not None
    assert match in problem


def test_restore_source_allows_default_optimizer_on_both_sides(tmp_path: Path) -> None:
    assert smoke._check_restore_source(_ckpt(tmp_path, ""), RECIPE) is None


def test_restore_source_refuses_non_checkpoint_dirs(tmp_path: Path) -> None:
    problem = smoke._check_restore_source(_ckpt(tmp_path, FUSED, parts=("model",)), RECIPE + FUSED)
    assert problem is not None
    assert "optim, step_scheduler.pt" in problem


def test_restored_tags_name_parent_run(tmp_path: Path) -> None:
    ckpt = tmp_path / "outputs" / "runs" / "parent-run" / "checkpoints" / "epoch_0_step_1"
    ckpt.mkdir(parents=True)
    tags = smoke._restored_tags(tmp_path, Path("outputs/runs/parent-run/checkpoints/epoch_0_step_1"))
    assert tags == ["resumed", "restored_from:parent-run/epoch_0_step_1"]
    assert smoke._restored_tags(tmp_path, None) == []
