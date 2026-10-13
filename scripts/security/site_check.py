"""Static-site security gate (M2-8): ``uv run python scripts/security/site_check.py <dist> [--csp-from-headers FILE]``.

Scans a built static site (``site/dist``) and fails when a page would load anything from another origin, embed a
known tracker, run inline script that the page's CSP does not allow by hash, ship a weak CSP, open ``_blank`` links
without ``noopener``, ship a font without its license file, or carry an NVIDIA logo file. Unfilled placeholders
(``TODO_MARKS``) only warn unless ``--publish``. Spec: ``.team/security-reviews/2026-10-13-m2-site-security-spec.md``
§6. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

TRACKER_HOSTS = (
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "googlesyndication.com",
    "plausible.io",
    "umami.is",
    "segment.com",
    "segment.io",
    "hotjar.com",
    "mixpanel.com",
    "posthog.com",
    "clarity.ms",
    "facebook.net",
    "facebook.com/tr",
    "connect.facebook.net",
    "cloudflareinsights.com",
    "vercel-insights.com",
    "va.vercel-scripts.com",
    "sentry.io",
    "fullstory.com",
    "amplitude.com",
    "intercom.io",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
)
# attributes that make the browser fetch something (navigation links in <a href> are allowed)
RESOURCE_ATTRS = {
    "script": ("src",),
    "img": ("src", "srcset"),
    "source": ("src", "srcset"),
    "video": ("src", "poster"),
    "audio": ("src",),
    "track": ("src",),
    "iframe": ("src",),
    "embed": ("src",),
    "object": ("data",),
    "input": ("src",),
    "image": ("href",),
    "use": ("href",),
}
FETCHING_LINK_RELS = {"stylesheet", "preload", "prefetch", "modulepreload", "icon", "apple-touch-icon", "manifest"}
FONT_SUFFIXES = {".woff2", ".woff", ".ttf", ".otf"}
FONT_LICENSE = re.compile(r"^(OFL|LICENSE|COPYING)[^/]*$", re.IGNORECASE)
CSS_URL = re.compile(r"""(?:@import\s+(?:url\()?|url\()\s*['"]?([^'")\s]+)""", re.IGNORECASE)
TODO_MARKS = ("{{TODO_USER", "{{TODO_SITE", "{{PRODUCT_NAME}}")
FORBIDDEN_SOURCES = ("'unsafe-eval'",)


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.rule}: {self.detail}"


def is_remote(url: str) -> bool:
    """True when ``url`` would be fetched from another origin (absolute http(s) or protocol-relative)."""
    url = url.strip()
    if url.startswith("//"):
        return True
    return urlsplit(url).scheme.lower() in {"http", "https", "ftp", "ws", "wss"}


def tracker_in(text: str) -> str | None:
    low = text.lower()
    return next((h for h in TRACKER_HOSTS if h in low), None)


def _srcset_urls(value: str) -> list[str]:
    return [part.strip().split()[0] for part in value.split(",") if part.strip()]


def csp_hash(script: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"


def parse_csp(policy: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            out.setdefault(tokens[0].lower(), tokens[1:])
    return out


def csp_problems(policy: str) -> list[str]:
    """Weaknesses of a CSP string per spec §2 (empty list = acceptable)."""
    d = parse_csp(policy)
    problems: list[str] = []
    script = d.get("script-src", d.get("default-src"))
    if script is None:
        problems.append("no script-src/default-src")
    else:
        if "'unsafe-inline'" in script:
            problems.append("script-src allows 'unsafe-inline'")
        if "*" in script or any(is_remote(s) or s in {"https:", "http:", "data:"} for s in script):
            problems.append("script-src allows other origins")
    for directive, sources in d.items():
        problems += [f"{directive} allows {s}" for s in sources if s in FORBIDDEN_SOURCES]
    if "'unsafe-inline'" in d.get("style-src-elem", d.get("style-src", [])):
        problems.append("style-src allows 'unsafe-inline' (only style-src-attr may)")
    if d.get("object-src", d.get("default-src")) != ["'none'"]:
        problems.append("object-src is not 'none'")
    if d.get("base-uri") not in (["'none'"], ["'self'"]):  # base-uri does not fall back to default-src
        problems.append("base-uri is not 'none' or 'self'")
    return problems


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.resources: list[tuple[str, str]] = []  # (tag.attr, url)
        self.csp: list[str] = []
        self.inline_scripts: list[str] = []
        self.inline_styles: list[str] = []
        self.blank_links: list[str] = []
        self._in: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        for attr in RESOURCE_ATTRS.get(tag, ()):
            if attr in a:
                urls = _srcset_urls(a[attr]) if attr == "srcset" else [a[attr]]
                self.resources += [(f"{tag}.{attr}", u) for u in urls]
        if tag == "link":
            rels = set(a.get("rel", "").lower().split())
            if rels & FETCHING_LINK_RELS and "href" in a:
                self.resources.append(("link.href", a["href"]))
            if "imagesrcset" in a:
                self.resources += [("link.imagesrcset", u) for u in _srcset_urls(a["imagesrcset"])]
        if tag == "meta" and a.get("http-equiv", "").lower() == "content-security-policy":
            self.csp.append(a.get("content", ""))
        if tag == "a" and a.get("target", "").lower() == "_blank" and "noopener" not in a.get("rel", "").lower():
            self.blank_links.append(a.get("href", ""))
        if "style" in a:
            self.resources += [("style-attr", u) for u in CSS_URL.findall(a["style"])]
        if (
            tag == "script"
            and "src" not in a
            and a.get("type", "").lower() not in {"application/json", "application/ld+json"}
        ):
            self._in, self._buf = "script", []
        elif tag == "style":
            self._in, self._buf = "style", []

    def handle_data(self, data: str) -> None:
        if self._in:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in == tag:
            text = "".join(self._buf)
            (self.inline_scripts if tag == "script" else self.inline_styles).append(text)
            self._in = None


def _script_sources(policy: str) -> list[str]:
    """Sources that govern inline <script>: script-src-elem, else script-src, else default-src (CSP3 fallback)."""
    d = parse_csp(policy)
    return d.get("script-src-elem", d.get("script-src", d.get("default-src", [])))


def check_html(rel: str, text: str, *, header_csp: str | None) -> list[Finding]:
    page = _Page()
    page.feed(text)
    found: list[Finding] = []
    for where, url in page.resources:
        if is_remote(url):
            found.append(Finding(rel, "REMOTE_RESOURCE", f"{where} -> {url}"))
    for style in page.inline_styles:
        found += [
            Finding(rel, "REMOTE_RESOURCE", f"<style> url -> {u}") for u in CSS_URL.findall(style) if is_remote(u)
        ]
    # prose may name a tracker ("we do not use segment.com"); only what the page loads or runs counts
    loaded = "\n".join([*(u for _, u in page.resources), *page.inline_scripts, *page.inline_styles])
    if (host := tracker_in(loaded)) is not None:
        found.append(Finding(rel, "TRACKER", host))
    found += [Finding(rel, "BLANK_WITHOUT_NOOPENER", href) for href in page.blank_links]
    policies = [*page.csp, *([header_csp] if header_csp else [])]
    if not policies:
        found.append(Finding(rel, "NO_CSP", "no <meta http-equiv=Content-Security-Policy> and no --csp-from-headers"))
    for policy in policies:
        found += [Finding(rel, "WEAK_CSP", p) for p in csp_problems(policy)]
    for script in page.inline_scripts:
        if script.strip() and not all(csp_hash(script) in _script_sources(p) for p in policies):
            found.append(Finding(rel, "INLINE_SCRIPT_NOT_HASHED", csp_hash(script)))
    return found


def check_css(rel: str, text: str) -> list[Finding]:
    found = [Finding(rel, "REMOTE_RESOURCE", f"css url -> {u}") for u in CSS_URL.findall(text) if is_remote(u)]
    if (host := tracker_in(text)) is not None:
        found.append(Finding(rel, "TRACKER", host))
    return found


def check_tree(dist: Path, *, header_csp: str | None = None) -> list[Finding]:
    found: list[Finding] = []
    files = sorted(p for p in dist.rglob("*") if p.is_file() and not p.is_symlink())
    for path in files:
        rel = path.relative_to(dist).as_posix()
        low = rel.lower()
        if "nvidia" in low and "logo" in low:
            found.append(Finding(rel, "NVIDIA_LOGO", "NVIDIA logos are not allowed (spec §4)"))
        if path.suffix.lower() in FONT_SUFFIXES and not any(FONT_LICENSE.match(p.name) for p in path.parent.iterdir()):
            found.append(Finding(rel, "FONT_WITHOUT_LICENSE", f"add OFL.txt/LICENSE next to it in {path.parent.name}/"))
        if path.suffix.lower() not in {".html", ".htm", ".css", ".js", ".mjs", ".json", ".xml", ".txt", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(m in text for m in TODO_MARKS):
            found.append(Finding(rel, "TODO_PLACEHOLDER", "unfilled placeholder (see TODO_MARKS)"))
        if path.suffix.lower() in {".html", ".htm"}:
            found += check_html(rel, text, header_csp=header_csp)
        elif path.suffix.lower() == ".css":
            found += check_css(rel, text)
        elif (host := tracker_in(text)) is not None:
            found.append(Finding(rel, "TRACKER", host))
    return found


def _headers_csp(path: Path) -> str | None:
    """First ``Content-Security-Policy:`` value in a Netlify/Cloudflare ``_headers`` file."""
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and key.strip().lower() == "content-security-policy":
            return value.strip()
    return None


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("dist", type=Path)
    ap.add_argument("--csp-from-headers", type=Path, help="_headers file whose CSP applies to every page")
    ap.add_argument("--publish", action="store_true", help="placeholders fail too (dev builds only warn)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not args.dist.is_dir():
        print(f"not a directory: {args.dist}", file=sys.stderr)
        return 2
    header_csp = _headers_csp(args.csp_from_headers) if args.csp_from_headers else None
    found = check_tree(args.dist, header_csp=header_csp)
    for f in found:
        print(("" if f.rule != "TODO_PLACEHOLDER" or args.publish else "warning: ") + str(f))
    found = [f for f in found if args.publish or f.rule != "TODO_PLACEHOLDER"]
    print(f"site_check: {len(found)} finding(s) in {args.dist}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
