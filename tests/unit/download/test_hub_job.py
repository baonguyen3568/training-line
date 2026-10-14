"""The download job (P018): filter, plan before fetching, budget, manifest; a fake hub, never the network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vlm_pipeline.common.download_rules import check_repo, name_ok, wanted
from vlm_pipeline.download import hub

REV = "a" * 40


@pytest.mark.parametrize(
    ("name", "kind", "ok"),
    [
        ("model.safetensors", "model", True),
        ("model-00001-of-00002.safetensors", "model", True),
        ("config.json", "model", True),
        ("tokenizer.json", "model", True),
        ("pytorch_model.bin", "model", False),
        ("modeling_x.py", "model", False),
        ("weights.gguf", "model", False),
        ("data/train-0000.parquet", "dataset", True),
        ("loader.py", "dataset", False),
        ("archive.zip", "dataset", False),
        ("../escape.json", "model", False),
        ("/abs.json", "model", False),
        ("random.dat", "model", False),
    ],
)
def test_the_shared_rules(name: str, kind: Any, ok: bool) -> None:
    assert wanted(name, kind) is ok


def test_python_only_when_listed_for_the_repo() -> None:
    assert wanted("modeling_x.py", "model", ["modeling_x.py"])
    assert not wanted("other.py", "model", ["modeling_x.py"])
    assert not wanted("loader.py", "dataset", ["loader.py"])  # a dataset never runs code


def test_repo_and_revision_shapes() -> None:
    check_repo("Qwen/Qwen3.5-0.8B", REV)
    for repo, rev in (("Qwen", REV), ("a/b/c", REV), ("Qwen/x", "main"), ("Qwen/x", "A" * 40)):
        with pytest.raises(ValueError, match="40-hex"):
            check_repo(repo, rev)
    assert not name_ok("a\nb")


def _sib(name: str, size: int, sha: str | None = None) -> Any:
    return SimpleNamespace(rfilename=name, size=size, lfs=SimpleNamespace(sha256=sha, size=size) if sha else None)


class _Hub:
    def __init__(self, siblings: list[Any], contents: dict[str, bytes]) -> None:
        self.siblings, self.contents, self.fetched = siblings, contents, []

    def model_info(self, repo: str, revision: str, files_metadata: bool) -> Any:
        assert files_metadata
        return SimpleNamespace(siblings=self.siblings)

    dataset_info = model_info

    def fetch(self, *, repo_id: str, filename: str, revision: str, repo_type: str, local_dir: str, **_: Any) -> str:
        self.fetched.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.contents[filename])
        return str(path)


def _run_dir(tmp_path: Path, **spec: Any) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    body = {"kind": "model", "repo": "org/m", "revision": REV, "max_bytes": 10_000, **spec}
    (run_dir / hub.SPEC_NAME).write_text(json.dumps(body))
    return run_dir


def test_a_model_is_planned_then_fetched_with_a_manifest(tmp_path: Path) -> None:
    contents = {"model.safetensors": b"w" * 100, "config.json": b"{}"}
    fake = _Hub([_sib("model.safetensors", 100, "f" * 64), _sib("config.json", 2), _sib("pytorch_model.bin", 50)],
                contents)  # fmt: skip
    run_dir = _run_dir(tmp_path)
    assert hub.run(run_dir, api=fake, fetch=fake.fetch) == 0
    plan = json.loads((run_dir / "stage" / "plan.json").read_text())
    assert [f["name"] for f in plan["files"]] == ["config.json", "model.safetensors"]  # .bin never planned
    assert plan["total_bytes"] == 102
    assert fake.fetched == ["config.json", "model.safetensors"]
    manifest = json.loads((run_dir / "stage" / "manifest.json").read_text())["files"]
    assert {m["name"]: m["size"] for m in manifest} == {"config.json": 2, "model.safetensors": 100}
    last = (run_dir / "stage" / "progress.jsonl").read_text().splitlines()[-1]
    assert json.loads(last)["bytes_done"] == 102


def test_over_the_budget_nothing_is_fetched(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = _Hub([_sib("model.safetensors", 20_000, "f" * 64)], {})
    run_dir = _run_dir(tmp_path)
    assert hub.run(run_dir, api=fake, fetch=fake.fetch) == hub.EXIT_OVER_BUDGET
    # PM 2026-10-14 (cord-v2): the job log says why, with both sizes, not just an exit code.
    err = capsys.readouterr().err
    assert "20000" in err.replace(",", "")
    assert "catalog size is too small" in err
    assert f"exit {hub.EXIT_OVER_BUDGET}" in err
    assert fake.fetched == []
    assert json.loads((run_dir / "stage" / "plan.json").read_text())["total_bytes"] == 20_000


def test_a_pickle_only_model_is_refused(tmp_path: Path) -> None:
    fake = _Hub([_sib("pytorch_model.bin", 10), _sib("config.json", 2)], {})
    assert hub.run(_run_dir(tmp_path), api=fake, fetch=fake.fetch) == hub.EXIT_NO_SAFETENSORS
    assert fake.fetched == []


@pytest.mark.parametrize("spec", [{"revision": "main"}, {"repo": "../x"}, {"max_bytes": 0}, {"kind": "space"}])
def test_a_bad_spec_runs_nothing(tmp_path: Path, spec: dict[str, Any]) -> None:
    fake = _Hub([], {})
    assert hub.run(_run_dir(tmp_path, **spec), api=fake, fetch=fake.fetch) == hub.EXIT_BAD_SPEC


def test_the_token_comes_from_the_mounted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = tmp_path / "hf_token"
    secret.write_text("hf_abc\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(secret))
    assert hub._token() == "hf_abc"
    monkeypatch.delenv("HF_TOKEN_PATH")
    assert hub._token() is None


def test_the_hubs_own_bookkeeping_never_reaches_the_host(tmp_path: Path) -> None:
    """PM 2026-10-14: hf_hub_download(local_dir=...) leaves .cache/huggingface/* next to the files; the host's exact
    check refused them (NET-003). The job removes that folder, so files/ is exactly the plan."""
    contents = {"model.safetensors": b"w" * 100, "config.json": b"{}"}
    fake = _Hub([_sib("model.safetensors", 100, "f" * 64), _sib("config.json", 2)], contents)

    def fetch(**kw: Any) -> str:
        path = fake.fetch(**kw)
        meta = Path(kw["local_dir"]) / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / f"{kw['filename']}.metadata").write_text("etag")
        (meta.parent / "CACHEDIR.TAG").write_text("x")
        return path

    run_dir = _run_dir(tmp_path)
    assert hub.run(run_dir, api=fake, fetch=fetch) == 0
    files = run_dir / "stage" / "files"
    assert sorted(p.relative_to(files).as_posix() for p in files.rglob("*") if p.is_file()) == [
        "config.json", "model.safetensors"]  # fmt: skip
    from vlm_pipeline.download.install import verify

    plan = json.loads((run_dir / "stage" / "plan.json").read_text())["files"]
    for f in plan:  # the fake hub's hashes: make them the content's so the host check can run end to end
        f["sha256"] = hashlib.sha256(contents[f["name"]]).hexdigest() if f["sha256"] else None
        f["git_sha1"] = None if f["sha256"] else _git_sha1(contents[f["name"]])
    (run_dir / "stage" / "plan.json").write_text(json.dumps({"files": plan}))
    assert set(verify(run_dir / "stage", kind="model", pins={}, name="m")) == {"config.json", "model.safetensors"}


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()  # noqa: S324 - git's id
