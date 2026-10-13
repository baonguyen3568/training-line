"""M2-8 static-site gate (spec .team/security-reviews/2026-10-13-m2-site-security-spec.md §6)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "security" / "site_check.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("site_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its module while the class body runs
    spec.loader.exec_module(module)
    return module


_sc = _load()
check_tree, csp_hash, csp_problems, main = _sc.check_tree, _sc.csp_hash, _sc.csp_problems, _sc.main

GOOD_CSP = (
    "default-src 'none'; script-src 'self' {h}; style-src 'self'; img-src 'self' data:; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'"
)


def _page(body: str, *, csp: str | None = GOOD_CSP, inline: str = "") -> str:
    meta = f'<meta http-equiv="Content-Security-Policy" content="{csp.format(h=csp_hash(inline))}">' if csp else ""
    script = f"<script>{inline}</script>" if inline else ""
    return f"<!doctype html><html><head>{meta}{script}</head><body>{body}</body></html>"


def _site(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def _rules(tmp_path: Path, files: dict[str, str]) -> set[str]:
    return {f.rule for f in check_tree(_site(tmp_path, files))}


def test_clean_site_passes(tmp_path: Path) -> None:
    body = (
        '<img src="/a.png" srcset="/a.png 1x, /a@2x.png 2x"><a href="https://huggingface.co" target="_blank" '
        'rel="noopener noreferrer">HF</a><link rel="stylesheet" href="/s.css">'
    )
    files = {
        "index.html": _page(body, inline="document.documentElement.dataset.theme='dark'"),
        "s.css": "@font-face{src:url(/fonts/inter/inter.woff2)}",
        "fonts/inter/inter.woff2": "x",
        "fonts/inter/OFL.txt": "SIL OFL",
    }
    assert _rules(tmp_path, files) == set()
    assert main([str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "body",
    [
        '<script src="https://cdn.jsdelivr.net/npm/x@1/x.js"></script>',
        '<link rel="stylesheet" href="//cdn.example.com/x.css">',
        '<img srcset="/a.png 1x, https://img.example.com/b.png 2x">',
        '<video poster="https://x.example/p.jpg"></video>',
        '<iframe src="https://www.youtube.com/embed/x"></iframe>',
        "<div style=\"background:url('https://x.example/bg.png')\"></div>",
        "<style>@import url(https://x.example/a.css);</style>",
    ],
)
def test_remote_resources_fail(tmp_path: Path, body: str) -> None:
    assert "REMOTE_RESOURCE" in _rules(tmp_path, {"index.html": _page(body)})


def test_navigation_links_are_not_resources(tmp_path: Path) -> None:
    body = '<a href="https://build.nvidia.com/spark" rel="noreferrer">NVIDIA</a>'
    assert _rules(tmp_path, {"index.html": _page(body)}) == set()


def test_trackers_fonts_cdn_and_css(tmp_path: Path) -> None:
    files = {
        "index.html": _page("<p>x</p>"),
        "a.js": "fetch('https://plausible.io/api/event')",
        "s.css": "@import url('https://fonts.googleapis.com/css2?family=Inter');",
    }
    found = check_tree(_site(tmp_path, files))
    assert {(f.path, f.rule) for f in found} >= {
        ("a.js", "TRACKER"),
        ("s.css", "REMOTE_RESOURCE"),
        ("s.css", "TRACKER"),
    }


def test_inline_script_needs_matching_hash(tmp_path: Path) -> None:
    html = _page("<script>alert(1)</script>", inline="ok()")
    assert "INLINE_SCRIPT_NOT_HASHED" in _rules(tmp_path, {"index.html": html})


def test_json_ld_is_not_script(tmp_path: Path) -> None:
    html = _page('<script type="application/ld+json">{"@type":"Organization"}</script>')
    assert _rules(tmp_path, {"index.html": html}) == set()


@pytest.mark.parametrize(
    ("policy", "needle"),
    [
        ("default-src 'self'; script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'", "unsafe-inline"),
        ("default-src 'none'; script-src 'self' 'unsafe-eval'; object-src 'none'; base-uri 'none'", "unsafe-eval"),
        ("default-src 'none'; script-src 'self' https:; object-src 'none'; base-uri 'none'", "other origins"),
        ("default-src 'none'; script-src 'self'; object-src 'none'", "base-uri"),
        ("default-src 'self'; script-src 'self'; base-uri 'none'", "object-src"),
        (
            "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; object-src 'none'; base-uri 'none'",
            "style",
        ),
    ],
)
def test_weak_csp(policy: str, needle: str) -> None:
    assert any(needle in p for p in csp_problems(policy))


def test_style_attr_may_be_unsafe_inline() -> None:
    policy = "default-src 'none'; script-src 'self'; style-src-attr 'unsafe-inline'; object-src 'none'; base-uri 'none'"
    assert csp_problems(policy) == []


def test_missing_csp_unless_headers_file(tmp_path: Path) -> None:
    site = _site(tmp_path / "dist", {"index.html": _page("<p>x</p>", csp=None)})
    assert {f.rule for f in check_tree(site)} == {"NO_CSP"}
    headers = tmp_path / "_headers"
    headers.write_text("/*\n  Content-Security-Policy: " + GOOD_CSP.format(h="") + "\n", encoding="utf-8")
    assert main([str(site), "--csp-from-headers", str(headers)]) == 0


def test_blank_link_todo_font_and_logo(tmp_path: Path) -> None:
    files = {
        "index.html": _page('<a href="/x" target="_blank">x</a>'),
        "legal/privacy/index.html": _page("<p>Liên hệ: {{TODO_USER: email}}</p>"),
        "fonts/x/x.woff2": "x",
        "img/NVIDIA-logo.svg": "<svg/>",
    }
    assert _rules(tmp_path, files) == {
        "BLANK_WITHOUT_NOOPENER",
        "TODO_PLACEHOLDER",
        "FONT_WITHOUT_LICENSE",
        "NVIDIA_LOGO",
    }


def test_not_a_directory(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing")]) == 2


def test_placeholders_fail_only_with_publish(tmp_path: Path) -> None:
    site = _site(tmp_path, {"index.html": _page("<p>{{TODO_USER: email}}</p>")})
    assert main([str(site)]) == 0
    assert main([str(site), "--publish"]) == 1


def test_prose_naming_a_tracker_is_not_a_tracker(tmp_path: Path) -> None:
    html = _page("<p>Chúng tôi không dùng segment.com hay plausible.io.</p>")
    assert _rules(tmp_path, {"index.html": html}) == set()
    html = _page('<img src="/p.gif?u=https://www.google-analytics.com/collect">')
    assert "TRACKER" in _rules(tmp_path / "b", {"index.html": html})


def test_inline_hash_must_be_allowed_by_every_policy(tmp_path: Path) -> None:
    inline = "ok()"
    meta2 = "default-src 'none'; script-src 'self'; object-src 'none'; base-uri 'none'"
    html = _page("", inline=inline).replace(
        "<script>", f'<meta http-equiv="Content-Security-Policy" content="{meta2}"><script>', 1
    )
    assert "INLINE_SCRIPT_NOT_HASHED" in _rules(tmp_path, {"index.html": html})


def test_inline_hash_in_default_src_counts(tmp_path: Path) -> None:
    inline = "ok()"
    csp = f"default-src 'self' {csp_hash(inline)}; object-src 'none'; base-uri 'none'"
    html = _page("", csp=csp, inline=inline)
    assert _rules(tmp_path, {"index.html": html}) == set()
