"""Unit tests for ``vlm_pipeline.artifacts.runs_index`` (list_runs + metrics paging)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_pipeline.artifacts.runs_index import list_runs, read_metrics_page
from vlm_pipeline.artifacts.testing import FakeRunSpec, make_fake_run, seed_demo_runs
from vlm_pipeline.common.runinfo import RunStatus


def test_list_runs_newest_first_and_skips_broken(tmp_path: Path) -> None:
    runs = tmp_path / "outputs" / "runs"
    seed_demo_runs(runs)
    (runs / "broken").mkdir()
    (runs / "broken" / "run.json").write_text("{not json")
    (runs / "no-manifest").mkdir()
    got = list_runs(tmp_path)
    assert "broken" not in {m.run_id for m in got}
    assert len(got) == 9
    starts = [m.started_at for m in got]
    assert starts == sorted(starts, reverse=True)
    assert {m.status for m in got} == set(RunStatus)


def test_list_runs_without_runs_dir(tmp_path: Path) -> None:
    assert list_runs(tmp_path) == []


def _run(tmp_path: Path, n: int = 25) -> Path:
    return make_fake_run(tmp_path, FakeRunSpec("r", n_steps=n, checkpoint_every=0))


def test_paging_covers_every_line_exactly_once(tmp_path: Path) -> None:
    run = _run(tmp_path)
    steps: list[object] = []
    cursor: str | None = None
    while True:
        page = read_metrics_page(run, cursor, limit=7)
        steps += [p["step"] for p in page.points]
        cursor = page.next_cursor
        if not page.has_more:
            break
    assert steps == list(range(1, 26))
    tail = read_metrics_page(run, cursor, limit=7)  # tailing a finished file returns nothing new
    assert tail.points == []
    assert not tail.has_more


def test_half_written_last_line_is_not_returned(tmp_path: Path) -> None:
    run = _run(tmp_path, n=3)
    with (run / "metrics.jsonl").open("a") as fh:
        fh.write('{"schema_version":1,"step":4')  # writer still flushing
    page = read_metrics_page(run, None, limit=100)
    assert [p["step"] for p in page.points] == [1, 2, 3]
    with (run / "metrics.jsonl").open("a") as fh:
        fh.write(',"ts":1.0}\n')
    nxt = read_metrics_page(run, page.next_cursor, limit=100)
    assert [p["step"] for p in nxt.points] == [4]


def test_downsample_keeps_ends_and_extremes(tmp_path: Path) -> None:
    run = _run(tmp_path, n=200)
    page = read_metrics_page(run, None, limit=10, downsample_to=20)
    steps = [p["step"] for p in page.points]
    assert page.downsampled
    assert not page.has_more
    assert steps[0] == 1
    assert steps[-1] == 200
    assert len(steps) <= 20
    assert steps == sorted(steps)
    small = read_metrics_page(run, None, limit=10, downsample_to=500)
    assert not small.downsampled
    assert len(small.points) == 200


def test_missing_metrics_file(tmp_path: Path) -> None:
    page = read_metrics_page(tmp_path, None, limit=10)
    assert page.points == []
    assert not page.has_more


@pytest.mark.parametrize(
    ("after", "limit", "downsample_to"),
    [("abc", 10, None), ("-1", 10, None), (None, 0, None), (None, 10_001, None), ("5", 10, 50), (None, 10, 1)],
)
def test_bad_arguments(tmp_path: Path, after: str | None, limit: int, downsample_to: int | None) -> None:
    run = _run(tmp_path, n=3)
    with pytest.raises(ValueError):  # noqa: PT011 - several distinct messages, all user input errors
        read_metrics_page(run, after, limit, downsample_to)
