"""Inline Lucide icons into the mock: index.src.html {{i:name}} -> <svg>, writes index.html (no runtime fetch)."""
import re
from pathlib import Path

here = Path(__file__).parent


def icon(name: str) -> str:
    svg = (here / "icons" / f"{name}.svg").read_text()
    svg = re.sub(r"\s+", " ", svg).strip()
    svg = re.sub(r'width="24" height="24"', 'width="1em" height="1em" aria-hidden="true" focusable="false"', svg)
    return svg.replace("<svg ", '<svg class="ic" ', 1)


src = (here / "index.src.html").read_text()
(here / "index.html").write_text(re.sub(r"\{\{i:([a-z0-9-]+)\}\}", lambda m: icon(m.group(1)), src))
print("ok")
