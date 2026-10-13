---
version: 1
slug: "frontend-src-app-tsx"
primary_target: "frontend/src/App.tsx"
related_targets: ["frontend/src"]
---

# Surface brief: dashboard / editor (NTC Training)

Scope: the app editor in `frontend/`, built on its own route until the user approves (7870 stays untouched while the user tests). Mode: Operate.
Task: build or open a workflow, wire nodes, check, run, and watch per-node state; Advanced users open subgraphs and engine params; Settings; reference pipeline gallery.
Constraints: the whole toolset is visible in one 1440×900 screen; icon-first with tooltips/popovers; no stray helper text; dark-first; WCAG AA; status labels in text tokens, status ink for marks.
User decision 2026-10-13: the D4 frame, pushed further icon-first. The left dock is an icon strip with a library flyout; bottom tabs, toolbar and the Simple/Advanced toggle are icons; the inspector keeps param labels, with explanations behind "?".

## Direction contract
THESIS: An Unreal-clean editor where tools are glyphs and the canvas gets the space. It refuses a text-heavy sidebar and helper prose.
OWN-WORLD: near-black elevations, hairlines, indigo #6e56cf for selection and the Run action only, typed ports ●■◆ from tokens, status ink for icons and borders, Be Vietnam Pro 12–14px UI, Lucide 16/20px icons.
STORY: The user sees every tool at once, opens the library flyout, drags a node, connects it, sees the check bar go green, presses Run, and watches each node's state. Advanced users expand a subgraph in place.
FIRST VIEWPORT: 48px top toolbar, icon groups left to right: file, edit, view, mode icon, GPU/RAM/disk chips, labelled Run at right. 48px left icon strip; flyout library 280px on demand. The canvas fills the centre. A 320px inspector on the right. A 32px bottom dock with icon tabs that expands to 180px.
FORM: canon editor (Unreal Editor / Figma grammar), seed key 00ab95ce.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
