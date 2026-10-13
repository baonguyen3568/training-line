---
name: NTC-Training
description: No-code node-graph platform for training vision-language models; enterprise-dark, product-as-hero (direction D4 "Chuẩn doanh nghiệp").
colors:
  accent-brand: "#6e56cf"
  accent-focus: "#b3a6ff"
  action-primary-hover: "#5f48c6"
  action-primary-fg: "#ffffff"
  page: "#0d0d0d"
  canvas-background: "#121211"
  canvas-lane-band: "#171716"
  surface: "#1a1a19"
  canvas-placeholder: "#222220"
  grid: "#2c2c2a"
  axis: "#383835"
  canvas-grid-dot: "#262624"
  node-border: "rgba(255,255,255,0.14)"
  text-primary: "#f0efec"
  text-secondary: "#c3c2b7"
  text-muted: "#898781"
  port-dataset: "#199e70"
  port-model: "#3987e5"
  port-report: "#d95926"
  status-good: "#0ca30c"
  status-warning: "#fab219"
  status-serious: "#ec835a"
  status-critical: "#d03b3b"
  status-ink-good: "#0ca30c"
  status-ink-warning: "#fab219"
  status-ink-serious: "#ec835a"
  status-ink-critical: "#d03b3b"
  delta-better: "#0ca30c"
  delta-worse: "#e66767"
typography:
  display:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "clamp(2.375rem, 4.2vw, 3.75rem)"
    fontWeight: 700
    lineHeight: 1.06
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "clamp(1.875rem, 2.8vw, 2.5rem)"
    fontWeight: 700
    lineHeight: 1.12
    letterSpacing: "-0.03em"
  title:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  body-table:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Be Vietnam Pro, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.3
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
    fontFeature: "tnum"
  stat:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "28px"
    fontWeight: 500
    lineHeight: 1.1
    fontFeature: "tnum"
rounded:
  sm: "4px"
  md: "8px"
  full: "999px"
spacing:
  "4": "4px"
  "8": "8px"
  "12": "12px"
  "16": "16px"
  "24": "24px"
  "32": "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent-brand}"
    textColor: "{colors.action-primary-fg}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  button-primary-hover:
    backgroundColor: "{colors.action-primary-hover}"
    textColor: "{colors.action-primary-fg}"
  button-primary-sm:
    backgroundColor: "{colors.accent-brand}"
    textColor: "{colors.action-primary-fg}"
    rounded: "{rounded.md}"
    padding: "0 12px"
    height: "32px"
  button-ghost:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  icon-button:
    backgroundColor: "transparent"
    textColor: "{colors.text-secondary}"
    rounded: "{rounded.md}"
    size: "32px"
  icon-button-pressed:
    backgroundColor: "{colors.canvas-placeholder}"
    textColor: "{colors.text-primary}"
  input:
    backgroundColor: "{colors.page}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-table}"
    rounded: "{rounded.md}"
    padding: "0 10px"
    height: "32px"
  tag:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    padding: "3px 8px"
  count-badge:
    backgroundColor: "{colors.canvas-placeholder}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.mono}"
    rounded: "{rounded.sm}"
    padding: "0 5px"
  node:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    width: "240px"
  popover:
    backgroundColor: "{colors.canvas-placeholder}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body-table}"
    rounded: "{rounded.md}"
    padding: "12px 14px"
  toolbar:
    backgroundColor: "{colors.surface}"
    height: "48px"
  icon-strip:
    backgroundColor: "{colors.surface}"
    width: "48px"
---

# Design System: NTC-Training

<!-- Source of truth for every value: design/tokens/{base,color.light,color.dark}.tokens.json (DTCG, built into frontend/ and site/ by build-tokens). The frontmatter above mirrors the DARK token set, which is the default mode. Light values live in color.light.tokens.json under the same names. If this file and the tokens disagree, the tokens win; fix this file. Derived from the shipped D4 mock design/m2/directions/chuan/index.html (views landing, dashboard, gallery, settings), 2026-10-13. -->

## Overview

**Creative North Star: "The Standard Instrument"**

NTC-Training plays the enterprise category straight. It has no illustrated hero, no metaphor and no glow. The working product is the hero: on the site the real editor is embedded live, wired up and pre-checked, and on the dashboard every tool is visible in one 1440×900 screen. Each detail is held to a 1px standard. The finish benchmarks are Linear, Vercel and Modal for the site, and Unreal Editor 5 and Figma for the editor. The system's identity comes from precision, the typed-port grammar (colour + shape + label) and the indigo NTC mark. It does not come from decoration.

The world is dark-first. Near-black neutral elevations are separated by 1px hairlines. One indigo accent is reserved for the primary action, selection and focus. Port and status colours carry the only other saturation, and they always arrive with a shape, an icon or text. The site (Persuade) and the editor (Operate) share the same header grammar, port shapes and "?" popover mechanism. The site breathes in large type and long sections. The editor is dense at 12–14px and icon-first, with Lucide glyphs and tooltips instead of words.

The voice of the interface is Vietnamese with full diacritics. Secondary explanation always sits behind a "?" popover, never in the body. Mono type is reserved for commands and numbers.

**Key Characteristics:**
- Dark, cool-neutral tonal elevations with 1px hairline borders; flat at rest.
- One indigo accent (#6e56cf) for primary action, selection and focus only; never NVIDIA green.
- Typed ports: colour + shape (● ■ ◆ ▲) + label, always together.
- Icon-first editor chrome (Lucide), labelled only where a label is truly needed: inspector params and the Run button.
- Be Vietnam Pro everywhere; JetBrains Mono only for commands and numbers.
- WCAG 2.2 AA throughout; `prefers-reduced-motion` shows the final state with no animation.

## Colors

A near-black neutral field, one indigo voice and three CVD-safe port hues; everything else is grey.

### Primary
- **NTC Indigo** (`accent-brand`): the primary CTA ("Tải về & cài đặt", "Chạy"), the selected segment, toggles in the on position, and the NTC mark tile. White text on it reaches 5.4:1. Hover darkens to `action-primary-hover` (6.5:1); it never lightens.
- **Indigo Light** (`accent-focus`, dark mode): focus rings (2px, 2px offset), node selection ring, the active-item inset bar (2px) in the icon strip, settings nav and library results, the active bottom-tab underline, and accent-coloured links/text on dark. Light mode uses `#6e56cf` for focus.

### Secondary: port families (data types, not decoration)
- **Data Aqua** (`port-dataset`, ■ square): Dataset ports and wires.
- **Model Blue** (`port-model`, ● circle): Model and Checkpoint ports and wires. It is also the `node_state.running` ink and the progress-bar fill.
- **Report Orange** (`port-report`, ◆ diamond): Eval/Bench report ports.
- Endpoint (▲ triangle) uses `text-secondary`; plain values (dot) use `text-muted`. Subtypes (Adapter half-circle, QuantizedModel double-circle, ExportArtifact hexagon, BenchReport hollow diamond) are told apart by shape and label, never by a new hue (`base.tokens.json` `shape.port`).

### Tertiary: status
- **Status** (`status-*`) and **Status Ink** (`status-ink-*`): good, warning, serious and critical. In dark mode the two sets are identical. In light mode `status_ink` is darker (for example warning `#bb8004`) so that marks reach ≥3:1. Node states map through `color.node_state.*` in the tokens (running → model blue, done → good, interrupted → warning, failed/blocked_policy → critical, others → text greys).
- **Delta** (`delta-better`/`delta-worse`): significant Δ vs baseline only, always with ▲/▼; when the CI contains 0, use `text-muted`.

### Neutral
- **Page Black** (`page`): site background, editor canvas surround, and input wells.
- **Canvas Black** (`canvas-background`) with **Grid Dot** (`canvas-grid-dot`, 20px grid) and **Lane Band** (`canvas-lane-band`): the node canvas.
- **Panel Graphite** (`surface`): toolbar, icon strip, flyout, inspector, dock, status bar, nodes and cards.
- **Raised Graphite** (`canvas-placeholder`): hover and pressed fills, popovers, tooltips and count badges.
- **Hairline** (`grid`), **Strong Hairline** (`axis`), **Node Border** (`node-border`): 1px dividers, control strokes and node outlines.
- **Ink** (`text-primary`), **Secondary Ink** (`text-secondary`), **Muted Ink** (`text-muted`, ≥4.5:1 on page and surface): three text steps. There is no fourth, dimmer step.

### Named Rules
**The Never-Colour-Alone Rule.** Every port is colour + shape (● Model, ■ Data, ◆ Report, ▲ Endpoint) + a text label. Every state is ink + Lucide icon + text. A screen rendered in greyscale must lose no meaning.

**The Ink-Not-Label Rule.** Status labels are set in `text-primary` / `text-secondary`. `status_ink` colours only icons, borders, bars, dots and LEDs. Status hues fail 4.5:1 as text in at least one mode.

**The One Indigo Rule.** Indigo marks the primary action, selection and focus, and nothing else. On the editor there is exactly one filled indigo control per screen: Run/Chạy.

**The No-Green-Brand Rule.** `#76B900` never appears as a brand colour, and nothing resembles the NVIDIA logo or eye. NVIDIA appears only as factual text ("Chạy trên NVIDIA NeMo AutoModel").

## Typography

**Display / Body Font:** Be Vietnam Pro (400, 600, 700; with system-ui, sans-serif)
**Mono Font:** JetBrains Mono (variable 100–800; with ui-monospace, monospace)

**Character:** Be Vietnam Pro is a geometric grotesk drawn for Vietnamese. Its stacked diacritics stay crisp from 12px UI labels to a 60px headline. JetBrains Mono, with tabular numerals, marks the things a user might copy or compare. Both are OFL-1.1, self-hosted (`site/public/fonts/`, `design/fonts/`), with no CDN.

### Hierarchy
- **Display** (700, 60px desktop / 38px at ≤760px, 1.06, -0.035em, `text-wrap: balance`, ≤21ch): landing hero only. The second sentence of the hero may step down to `text-secondary`.
- **Headline** (700, 40px / 30px mobile, 1.12, -0.03em, ≤16ch): site section heads.
- **Title** (600, 14–16px, 1.3): card titles, inspector node name, settings group heads, error-card heads.
- **Body** (400, 14px editor / 15–18px site lead, 1.55–1.6, ≤58ch): prose and popover text. The site lead is 18px in `text-secondary`.
- **Body-table** (400, 13px): settings rows, dock rows, library items, inputs.
- **Label** (500–600, 12px): inspector param labels, node titles, lane names, tags, tooltips.
- **Mono** (400–600, 12–13.5px, tabular): install command, GPU/RAM/disk chips, zoom %, numeric fields, counts, step numbers, model ids.
- **Stat** (mono, 28px): large numeric readouts.

### Named Rules
**The Mono-Means-Machine Rule.** JetBrains Mono is only for commands, paths, ids and numbers. It is never used for headings, labels or decoration.

**The Full-Diacritics Rule.** Every Vietnamese string carries full diacritics ("Đọc hoá đơn", not "Doc hoa don"). A face is accepted only if it renders stacked marks (ẩ, ồ, ặ) without clipping at the line-heights above.

## Layout

**Editor (dashboard), minimum width 1280px (`breakpoint.editor_min`), one-screen grid:**
- **Toolbar (48px top):** icon groups run left to right: NTC mark (32px) and workflow name with an unsaved dot | file (new/open/save) | edit (undo/redo) | view (tidy/fit/zoom). Then the Simple/Advanced icon toggle, the GPU/RAM/disk chips (icon + mono number), the reference-pipeline and settings icons, and finally Run/Stop. Run/Stop is the only control with text. Groups are separated by 1px × 20px dividers.
- **Icon strip (48px left):** library, step list, pipelines, search, with storage pinned at the bottom. Each icon is a 36px button with a tooltip to the right.
- **Library flyout (304px):** floats over the canvas beside the strip. It opens on `/` and closes on Esc, so the canvas keeps its full width.
- **Canvas** fills the centre. Its lane header (30px) carries numbered stage names.
- **Inspector (300px right):** a breadcrumb into subgraphs, then node head, param rows and estimates.
- **Bottom dock (176px):** a 34px row of icon-only tabs with mono count badges, above 30px rows. The **status bar** (26px) runs full width with a progress mini-bar.

**Settings and gallery** replace the canvas, inspector and dock while keeping the toolbar and strip. Settings uses a 252px section nav plus a content column (≤820px) of label/control rows (min 52px). The gallery is a 4-column card grid with a 16px gap and 32px gutters.

**Site:** the container is 1200px with 32px side gutters, and the sticky nav is 64px. The hero has 80px top padding and is followed by the live editor frame (aspect 1440/900) inside the first viewport. Sections are separated by 120–140px. The guard section is a 1fr + 460px two-column layout with an 80px gap. At ≤760px the layout collapses to a single column with 16px gutters, the nav moves into a menu button, and the CTA goes full width.

**Rhythm:** spacing comes from the 4px-based token scale (4/8/12/16/24/32). Editor panels pad 14px, and rows sit on 28–32px control heights.

### Named Rules
**The One-Screen Rule.** The whole editor toolset is visible at 1440×900 without scrolling chrome. A new tool earns a glyph in an existing group, not a new bar.

**The No-Stray-Text Rule.** Editor chrome carries no helper sentences. Explanations live behind "?" popovers, names live in tooltips. Only one dashed "minh hoạ" label is allowed next to illustrative numbers.

## Elevation & Depth

Depth is mostly tonal. Each layer steps up in lightness (page → canvas → surface → raised) and is separated by 1px hairlines, so surfaces are flat at rest. Shadows appear only on things that float above the plane: tooltips, popovers, the library flyout, the error card, nodes on the canvas, and the embedded editor frame on the site.

### Shadow Vocabulary
- **Float** (`box-shadow: 0 1px 0 #ffffff08 inset, 0 12px 32px -8px #000000b0, 0 2px 6px #00000080`): tooltips, popovers and error cards.
- **Node** (`box-shadow: 0 6px 18px -6px #000c`): canvas nodes at rest.
- **Selected node** (`box-shadow: 0 0 0 2px <accent-focus>, 0 10px 26px -8px #000`): the selection ring plus lift.
- **Flyout edge** (`box-shadow: 16px 0 40px -16px #000`): the library flyout over the canvas.
- **Showcase frame** (`box-shadow: 0 40px 120px -40px #000, 0 0 0 1px #00000080`): the live editor embedded in the landing hero only.

### Named Rules
**The Hairline-First Rule.** Separation is a 1px `grid` or `axis` line before it is ever a shadow. Shadows are reserved for floating layers.

## Shapes

Corners are gently rounded and consistent. Controls, inputs, cards, popovers and nodes use `rounded.md` (8px, also `node.radius`). Small badges (step numbers, counts, kbd, `mark`) use `rounded.sm` (4px). Tags, pills and switches are fully round. Ports are 10px glyphs (`node.port_size`) that straddle the node edge: circle, 1.5px-radius square, rotated-square diamond, or triangle. A hollow variant (ring on the node fill) marks the outline subtypes. Wires are 2px (`node.wire_width`) cubic curves in their port colour. A subgraph is a dashed indigo-tint frame (12px radius) with its own dashed header rule. A candidate or unreleased card uses a dashed border on a transparent fill. Dashed strokes always mean "container", "illustrative" or "not yet". They are never decorative.

## Components

### Buttons
- **Shape:** gently rounded (`rounded.md`), 40px default and 32px small, 600 weight, 8px icon gap.
- **Primary:** NTC Indigo with white text. Hover moves to `action-primary-hover`. Disabled is 45% opacity with a not-allowed cursor.
- **Ghost:** `surface` fill, 1px `axis` stroke, `text-primary`. Hover moves to raised graphite.
- **Run/Stop:** 32px, the toolbar's only labelled control. Run is primary indigo with a filled play glyph. Stop is a raised-graphite ghost with a 10px square.
- **Focus:** 2px `accent-focus` outline at 2px offset, on every interactive element.

### Icon buttons and the toolbar
- **Style:** a 32px square (36px in the strip, 26–30px in docks), transparent, `text-secondary` glyph, Lucide at 16px with stroke 1.75 (14px in dense rows).
- **Hover / pressed:** raised graphite fill with `text-primary`. Pressed toggles use `aria-pressed`. The strip's active item adds a 2px inset indigo-light bar on the left.
- **Tooltip:** every icon-only control has one. The tooltip is 12px/500 on raised graphite with a 1px `axis` stroke and Float shadow, and appears after a 250ms delay.
- **Simple/Advanced:** a single icon toggle button (`aria-pressed`), not a text segment.

### Chips, tags and counts
- **Resource chip:** 28px, mono 12.5px, icon + number, with a 7px LED dot in status ink. A dashed "minh hoạ" tag sits beside illustrative values.
- **Tag:** pill, 11px/500 `text-muted`, 1px `axis` stroke.
- **Count badge:** mono 10.5px on raised graphite, 4px radius. It is used on dock tabs, library groups and search results.

### Cards / Containers
- **Corner Style:** `rounded.md`.
- **Background:** `surface` on `page`, with a 1px `axis` stroke. Hover shifts the stroke to an indigo tint.
- **Shadow Strategy:** flat. The error card alone carries Float (see Elevation & Depth).
- **Internal Padding:** 14–18px.
- **Gallery card:** a mini-diagram thumbnail on a dot grid. The source is shown as text (no logos). It has a definition list (licence, difficulty, time) where unknowns read "xem nguồn", "chưa đánh giá" or "chưa có số đo" in `text-muted`. Actions sit at the bottom.

### Inputs / Fields
- **Style:** 32px, `page` well, 1px `axis` stroke, `rounded.md`, 13px text. Numeric fields are right-aligned mono, and steppers use 26px icon buttons.
- **Segmented control:** a page-well track with 2px padding. The selected segment is raised graphite + `text-primary`. Inspector presets that are the primary choice may use `action.selected_bg` indigo.
- **Switch:** 30×18 pill, indigo when on, graphite when off.
- **Focus:** the global indigo-light ring.
- **Locked (settings):** the row label steps down to `text-secondary`, with a lock icon button whose popover explains who locked it and why. Locked rows are read-only, not disabled-grey.
- **Secret:** masked `hf_••••••••` in a mono field plus a connection pill (icon + text).

### Inspector rows
Each param keeps its text label (12px/500 `text-secondary`), followed by a 20px "?" button that opens a popover. The control sits right-aligned, and no explanatory prose appears under the field. A breadcrumb (11.5px `text-muted`, chevron-separated) shows the path into a subgraph. Estimates (time, RAM, disk) are icon + label + value; an unknown value reads "chưa có số đo".

### Navigation
- **Site nav:** sticky 64px, translucent page black with backdrop blur and a 1px bottom hairline. It holds the wordmark (24px mark), five 14px `text-secondary` links (hover gives `text-primary` on `surface`), then "Tài liệu" ghost and the primary "Tải về & cài đặt". On mobile the links collapse into a menu button.
- **Settings nav:** 32px rows with icon + short label. The current row is raised graphite with the inset indigo-light bar and an indigo-light icon.
- **Bottom dock tabs:** icon-only with tooltips and count badges. The selected tab has `text-primary` and a 2px indigo-light underline.

### Node (signature component)
Nodes are `surface` with a 1px `node-border` stroke, `rounded.md` and `node.width`. The header row holds a 14px Lucide icon, a 12px/600 title and a mono step-number badge. Optional markers sit to its right: a wrench for "Advanced only" and stacked layers for "contains sub-steps". Below the header come a one-line summary in `text-muted`, then the port rows (inputs left, outputs right, 10px port glyphs on the edge, 11px labels). The state footer shows a state icon in ink + a text label in text greys, with a 3px progress bar in `port-model` while running. Running nodes take a `port-model` border. Nodes missing an input take a critical-ink border, and their missing port pulses (static under reduced motion). The selected node has the indigo-light ring.

### Subgraph frame
An expanded group is a dashed indigo-tint frame on a 5% indigo wash. Its 34px header shows icon, name, step badge, live state (icon + text) and a collapse button. Outer ports sit on the frame edge and show their names on tooltip. A collapsed group is an ordinary node with the stacked-layers marker.

### "?" Popover
A 20px circular ghost trigger in `text-muted`. The panel is ≤320px, raised graphite, 1px `axis` stroke, `rounded.md`, 13px/1.5 text and the Float shadow. An optional mono source line ("Nguồn: …") sits at the foot. This is the single mechanism for secondary text on the site, in docs and in the editor.

### Wordmark
The mark is an indigo rounded tile (8/32 radius) with a geometric white "N" (square caps, mitred). Beside it, "NTC" is set in 700 and "Training" in 400 `text-secondary`. It is inline SVG at 24px in the site nav and 32px in the editor toolbar.

### Motion
State changes use 120–150ms colour, opacity and transform transitions. A wire draws in 700ms on `cubic-bezier(.16,1,.3,1)`. Running wires carry a 1s linear dash flow. The landing hero has one choreographed moment: the editor scrolls in, the data wire connects, and the check bar turns "Sẵn sàng". Under `prefers-reduced-motion: reduce` all animation and transitions are removed and the final state is shown.

## Do's and Don'ts

### Do:
- **Do** pair every port colour with its shape (● Model, ■ Data, ◆ Report, ▲ Endpoint) and a text label, and every state colour with a Lucide icon and text.
- **Do** set status labels in `text-primary`/`text-secondary`, and use `status_ink` only for icons, borders, bars and dots.
- **Do** keep the editor icon-first: 48px toolbar, 48px left icon strip with a 304px library flyout (`/`, Esc), icon-only dock tabs with tooltips and counts, and an icon toggle for Simple/Advanced.
- **Do** keep param labels in the inspector, with their explanations behind a "?" popover.
- **Do** give every icon-only control a tooltip and an accessible name.
- **Do** use Be Vietnam Pro for all UI and prose, and JetBrains Mono only for commands and numbers. Both are self-hosted under OFL-1.1, and icons come from Lucide only.
- **Do** meet WCAG 2.2 AA: text ≥4.5:1, non-text marks ≥3:1, a visible 2px focus ring, and full keyboard reach. Honour `prefers-reduced-motion` by showing the end state.
- **Do** write Vietnamese with full diacritics, and show unknown numbers as "chưa có số đo" rather than inventing them.
- **Do** take every colour, radius, spacing and node dimension from `design/tokens/` through build-tokens.

### Don't:
- **Don't** use `#76B900` or any NVIDIA-green as a brand colour, and don't draw anything resembling the NVIDIA logo or eye.
- **Don't** set a status label's text in a status colour (green "Đã kết nối", red "Chưa đủ").
- **Don't** add a port hue. Only three pass CVD all-pairs, so subtypes differ by shape + label.
- **Don't** put helper sentences, captions or small notes in editor chrome. Use the "?" popover.
- **Don't** add labelled text buttons to the toolbar. Run/Stop is the only one.
- **Don't** use indigo for anything other than the primary action, selection and focus.
- **Don't** use mono for headings or labels, or load fonts and icons from a CDN.
- **Don't** pair a Lucide icon with a duplicate text glyph (an icon followed by "✓" or "✕").

### Known drift (mock vs tokens)
The D4 mock hard-codes its own palette and sizes. The tokens are normative, and these mock values must not be copied into product code:
- **Neutrals:** the mock uses a cool ramp (`#09090b / #0f1013 / #15171b / #1c1e23 / #24272d`, lines `#23262c / #30343b`, text `#ededf0 / #a7acb6 / #858b95`). The tokens use a warm-neutral dark set (`page #0d0d0d`, `canvas #121211`, `surface #1a1a19`, `placeholder #222220`, `grid #2c2c2a`, `axis #383835`, text `#f0efec / #c3c2b7 / #898781`). The endpoint port uses mock `#a7acb6` vs token `text.secondary`. Canvas dots are mock `#ffffff12` vs `grid_dot #262624`. The node border is mock `#30343b` vs `rgba(255,255,255,0.14)`. If the cooler cast is wanted, change the tokens first (token owner + PM), then this file.
- **Primary hover:** mock `#7d67d9` lightens the button and gives only 4.36:1 with white text, which fails AA. The token `#5f48c6` (6.5:1) is canonical.
- **Off-token status text colours:** the mock adds `--ok-tx #3fc43f` and `--crit-tx #f07070` and uses them as label text on the node "Chưa đủ" state, the error rows, the "✓ Đã kết nối" pill and the 6/6 count. This breaks the Ink-Not-Label Rule. These colours belong on icons only, and the labels go to text greys.
- **Radius:** mock controls use 6px and cards 10–12px. The tokens have only `radius.sm 4` / `radius.md 8`, so 8 is canonical. The subgraph frame's 12px radius is a one-off.
- **Node size:** mock nodes are 150px wide (small variant 11px text) with a ~30px header. The tokens are `node.width 240`, `node.header_height 32`.
- **Type below the scale:** mock editor text at 10–11.5px (port labels, node summary, lane names, badges, status bar). The smallest token is `font.size.caption 12`. The lane "empty" label `#5d626b` is 3.24:1, which fails AA.
- **Font family tokens:** resolved on 2026-10-13. `base.tokens.json` now leads with Be Vietnam Pro (UI) and JetBrains Mono (mono), with system stacks as fallbacks.
- **Panel widths vs the dashboard contract:** the build has a 304px flyout, a 300px inspector and a 176px dock. The surface brief said 280 / 320 / 180. The build values are recorded here.
