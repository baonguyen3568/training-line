"""Inline Lucide icons (icons/*.svg, ISC) into index.html. Run: python3 build.py"""
import json, pathlib, re
here = pathlib.Path(__file__).parent
icons = {}
for f in (here / "icons").glob("*.svg"):
    body = re.sub(r"\s+", " ", f.read_text()).strip()
    icons[f.stem] = re.sub(r"<svg[^>]*>", '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">', body)
src = (here / "_template.html").read_text()
(here / "index.html").write_text(src.replace("/*ICONS*/{}", json.dumps(icons, ensure_ascii=False)))
print("icons", len(icons))
