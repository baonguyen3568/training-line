"""G-3 send_report: the server-built bug report never carries credentials, host paths or user text."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.common.bug_report import REDACTED, ReportContext, build_report, redact
from vlm_pipeline.common.runinfo import RunManifest, RunStatus, write_manifest

FAKE_HF = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"  # credential SHAPE only, built at runtime (secret gate)


def test_credentials_and_assignments_are_redacted() -> None:
    assert FAKE_HF not in redact(f"token is {FAKE_HF} here")
    assert redact("HF_TOKEN=abc123 other=1") == f"HF_TOKEN={REDACTED} other=1"
    assert redact("db password: hunter2") == f"db password: {REDACTED}"
    assert redact("loss=0.5 lr=1e-4") == "loss=0.5 lr=1e-4"  # ordinary values stay


def test_host_paths_become_project_relative(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    assert redact(f"open {root}/outputs/runs/r/x.json failed", root) == "open outputs/runs/r/x.json failed"


def _run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "proj"
    run = root / "outputs" / "runs" / "20261013-1500-x-ab12"
    (run / "logs").mkdir(parents=True)
    manifest = RunManifest(run_id=run.name, name="x", stage="train", task="lora", status=RunStatus.FAILED,
                           error_code="MEM-002", exit_code=137, container_image="vlm-automodel:26.08-6")  # fmt: skip
    write_manifest(run, manifest)
    lines = [f"step {i}" for i in range(200)] + [f"export HF_TOKEN={FAKE_HF}", f"cfg at {root}/configs/a.yaml"]
    (run / "logs" / "stdout.log").write_text("\n".join(lines) + "\n")
    return root, run


def test_report_has_the_facts_and_no_secret(tmp_path: Path) -> None:
    root, run = _run(tmp_path)
    ctx = ReportContext(error_params={"need_gb": 40, "prompt": "my private receipt text"}, node_type="train.lora_vlm",
                        node_id="n4")  # fmt: skip
    text = build_report(root, run, ctx)
    assert "MEM-002" in text
    assert "exit_code=137" in text
    assert "vlm-automodel:26.08-6" in text
    assert "Node: train.lora_vlm (n4)" in text
    assert "step 199" in text
    assert "step 100" not in text  # only the last 80 lines
    assert FAKE_HF not in text
    assert str(root) not in text
    assert "configs/a.yaml" in text
    assert "my private receipt text" not in text
    assert "<23 chars, not copied>" in text


def test_a_run_dir_outside_the_project_is_refused(tmp_path: Path) -> None:
    root, _ = _run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        build_report(root, tmp_path)


def test_a_run_without_run_json_still_reports(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    run = root / "outputs" / "runs" / "empty"
    run.mkdir(parents=True)
    assert "Run: outputs/runs/empty" in build_report(root, run, ReportContext(error_code="SYS-000"))


@pytest.mark.parametrize("swap", ["file_link", "dir_link", "fifo"])
def test_a_swapped_log_is_never_read(tmp_path: Path, swap: str) -> None:
    """bao-security B1: the job owns its run dir; a link to a host secret must not end up in a public report."""
    import os

    root, run = _run(tmp_path)
    canary = tmp_path / "secrets.key"
    canary.write_text("CANARY-not-a-shaped-token\n")
    log = run / "logs" / "stdout.log"
    log.unlink()
    if swap == "file_link":
        log.symlink_to(canary)
    elif swap == "dir_link":
        (run / "logs").rmdir()
        (run / "logs").symlink_to(tmp_path)
        (tmp_path / "stdout.log").symlink_to(canary)
    else:
        os.mkfifo(log)  # must not block the report
    text = build_report(root, run)
    assert "CANARY" not in text
    assert "log: not readable" in text


def test_wider_patterns_and_key_blocks_are_redacted() -> None:
    from vlm_pipeline.common.bug_report import GUESSED, redact_lines

    aws = "AKIA" + "ABCDEFGHIJKLMNOP"
    lines = [
        f"key {aws}",
        "Authorization: Bearer abc.def",
        "pull https://user:pa55@example.com/x",
        "-----BEGIN RSA PRIVATE KEY-----",
        "MIIEow",
        "-----END RSA PRIVATE KEY-----",
        "blob " + "Zm9v" * 12,
        "image sha256:" + "a" * 64,
        "session_id=xyz",
    ]
    out = redact_lines(lines)
    joined = "\n".join(out)
    assert aws not in joined
    assert "abc.def" not in joined
    assert "pa55" not in joined
    assert "MIIEow" not in joined
    assert out[3] == f"{REDACTED} (private key block)"
    assert GUESSED in out[4]
    assert "sha256:" + "a" * 64 in joined  # digests stay: a fact, not a secret
    assert "xyz" not in joined
