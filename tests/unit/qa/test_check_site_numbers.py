from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "qa" / "check_site_numbers.py"


def _checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_site_numbers", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page(tmp_path: Path, body: str, sources: list[str], extra: str = "") -> Path:
    listed = "".join(f"\n  - {s}" for s in sources)
    page = tmp_path / "page.mdx"
    page.write_text(f"---\ntitle: t\nsources:{listed}\n{extra}---\nimport X from '~/x';\n\n{body}\n", encoding="utf-8")
    return page


def _unsourced(tmp_path: Path, body: str, source_text: str, extra: str = "") -> list[str]:
    (tmp_path / "bench.json").write_text(source_text, encoding="utf-8")
    result = _checker().check_page(_page(tmp_path, body, ["bench.json"], extra), tmp_path)
    return [u["number"] for u in result["unsourced"]]


def test_made_up_number_is_reported_with_its_line(tmp_path: Path) -> None:
    (tmp_path / "bench.json").write_text('{"field_f1": 71.65}', encoding="utf-8")
    result = _checker().check_page(_page(tmp_path, "F1 đạt 71,65.\n\nTốc độ 3000 token/s.", ["bench.json"]), tmp_path)
    assert [(u["number"], u["line"]) for u in result["unsourced"]] == [("3000", 10)]


def test_vietnamese_decimal_comma_and_json_float_match(tmp_path: Path) -> None:
    assert (
        _unsourced(tmp_path, "Model 0,94 tỉ tham số, trống **42 GB**, 128 GB.", '{"p": 0.94, "m": 42.0, "x": "128GB"}')
        == []
    )


def test_thousands_separator(tmp_path: Path) -> None:
    assert _unsourced(tmp_path, "Tối đa 1.024 token.", "max_new_tokens: 1024") == []


def test_code_links_tags_and_error_codes_are_not_claims(tmp_path: Path) -> None:
    body = 'Chạy `--max 4096`, xem [trang](/docs/p-777/), mã CFG-010, <Hint width="555">.\n```\n9999\n```'
    assert _unsourced(tmp_path, body, "{}") == []


def test_explicit_whitelist(tmp_path: Path) -> None:
    extra = 'numbers_without_source: ["24", "80"]\n'
    assert _unsourced(tmp_path, "Card rời 24 GB hoặc 80 GB, máy này 96 GB.", "{}", extra) == ["96"]


def test_missing_source_file_is_reported(tmp_path: Path) -> None:
    result = _checker().check_page(_page(tmp_path, "Không có số.", ["benchmarks/nope.json"]), tmp_path)
    assert result["missing_sources"] == ["benchmarks/nope.json"]
