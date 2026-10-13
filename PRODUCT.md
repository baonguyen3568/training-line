# Product

<!-- impeccable:product-schema 1 -->

> Source: decisions the user confirmed (CLAUDE.md "Tầm nhìn sản phẩm", ADR-003/009/012, BOARD M1/M2), relayed by the PM on 2026-10-13.
> Lines tagged **[suy luận]** are design inferences, not user decisions.

## Platform

web

## Users
- **Primary — "Lan", a non-coder domain specialist** (accountant, operations, applied researcher). Knows what the model should get better at (reading invoices, charts, documents), has a folder of images plus questions/answers, does not write Python, and does not know what LoRA or NVFP4 means. Job: within one session, open a template → point it at their data → press Run → see how much better the new model is → export something usable.
- **Secondary — "Minh", an ML engineer on the team.** Needs Advanced mode, every parameter, logs, run comparison, and workflow files to share.
- **Website visitors:** people deciding whether to install the app (Persuade), and users reading docs or policies (Read). **[suy luận]** Mostly Vietnamese-speaking technical decision makers and the non-coders they support.

## Product Purpose
An end-to-end application, installed once, that lets someone who cannot code train a vision-language model entirely through the UI: get and prepare data → set up → train/fine-tune → try the model → compress (quantize) → benchmark → export. The UI is a drag-and-drop node graph (in the spirit of ComfyUI and Unreal Blueprint, but for training models): each step is a node, wires make a workflow, and workflows are saved and shared as files. Success = Lan finishes the first template run without help and understands the result.

## Positioning
- A no-code node graph for **training** VLMs (not image generation), with errors **blocked before running**: typed ports, RAM/time/disk estimates, safe defaults, plain-language error cards with fixes. The point is never failing three hours into a training run.
- The engine underneath is NVIDIA's official stack (NeMo AutoModel, Megatron-Bridge, NeMo-RL, NeMo Curator, TensorRT Model Optimizer, TensorRT-LLM, NeMo Evaluator), tuned first for DGX Spark (GB10, 128 GB unified memory, aarch64).
- Reuse open source rather than reinvent it (React Flow, ideas from ComfyUI and LLaMA-Factory).

## Operating Context
- Runs on a **shared** DGX Spark: other services already use 70+ GB of RAM. Only one heavy GPU job runs at a time (GPU lock); the UI must explain "the machine is busy" without blaming the user.
- Ships with templates people open and run straight away, then edit:
  - the default is "Đọc hoá đơn → JSON" (Nemotron-Parse v1.1, full fine-tune on CORD-v2);
  - template 2 is image/video understanding on Cosmos-Reason2-2B (gated on Hugging Face);
  - template 3 is quantize + benchmark.
- Simple mode is the default (2–4 parameters with presets); Advanced shows everything.
- Third-party custom nodes are allowed from v1, sandboxed, with trust levels and permission warnings (ADR-009 A1).

## Capabilities and Constraints
- **Language:** Vietnamese by default (with full diacritics), English available. A single i18n source: `src/vlm_pipeline/i18n/{vi,en}.yaml`.
- **Free and open source,** Apache-2.0 (ADR-003). There is no paid tier; the "pricing" page means "Free & open source".
- **Website: 20–30 pages.**
  - Pages: a landing page with eye-catching animation, about, features, use cases, templates, multi-page docs, policies, contact, changelog, roadmap, FAQ, 404.
  - Secondary explanations live behind a "?" icon + popover, never stuffed into the body. Outer pages may be long.
- **Dashboard/editor:**
  - Clean in the way Unreal Editor is clean: the whole toolbar fits on one screen.
  - Icon-first with tooltip/popover; no text unless it is truly needed; no stray small notes.
  - Enterprise grade.
  - Built on its own route until the user approves it; the live app on 7870 stays untouched while the user is testing.
- **No trackers, analytics, or third-party CDNs.** Fonts are self-hosted (OFL-1.1); icons are Lucide/Phosphor/Tabler; every asset is logged in `site/ASSETS.md` (security spec M2-8).
- **Product name:** "NTC-Training" (user confirmed 2026-10-13). Wordmark may read "NTC Training"; slug/repo `ntc-training`. "NTC" is the primary identity.
- **Open:** site host (U9), policy details U7–U10.

## Brand Commitments
- No affiliation with NVIDIA is implied:
  - no NVIDIA logo or "eye";
  - no #76B900 as the primary brand colour;
  - no "Powered by / Partner / Certified" badges;
  - the product name must not contain NVIDIA, NeMo, Nemotron, Cosmos, DGX, CUDA, or TensorRT.
- Factual nominative use is fine ("Chạy trên NVIDIA NeMo AutoModel", "Tối ưu cho DGX Spark").
- "Built on NVIDIA Cosmos" is mandatory wherever the Cosmos template is presented, and on the licenses page (ADR-012).
- Existing app design decisions remain binding until a new direction is chosen (`design/tokens/`, P010):
  - port families use colour + shape + label;
  - status colours always come with icon + text;
  - nothing carries meaning through colour alone.

## Evidence on Hand
- **Real:** benchmark/perf JSON in `benchmarks/perf/`, ADRs, the model catalog `configs/catalog/models.yaml`, i18n copy, app screenshots (with CORD-v2 sample data, CC BY 4.0 attribution).
- **Absent, never to be fabricated:** customers, logos, testimonials, user counts, star counts, press, accuracy numbers that are not in `benchmarks/`.
- Numbers still being measured show "chưa có số đo".

## Product Principles
1. **Mở là chạy** (open it and it runs): every template runs on safe defaults with a small sample dataset.
2. **Chặn lỗi trước khi chạy** (block errors before running): validate wiring, memory, and disk up front; never fail hours in.
3. **Ngôn ngữ của người dùng** (the user's language): plain Vietnamese first, technical terms as secondary annotation behind "?".
4. **Trung thực về số liệu** (honest numbers): estimates show their source (measured / formula / declared / unknown); nothing is invented.
5. **Không chỉ dùng màu** (never colour alone): meaning always comes with shape, icon, and text.

## Accessibility & Inclusion
WCAG 2.2 AA (contrast ≥ 4.5:1 for text, visible focus, keyboard reachability, `prefers-reduced-motion` respected). Vietnamese diacritics must render correctly in every chosen typeface. **[suy luận]** Users include non-native English readers, so the UI is icon-first but never icon-only for critical actions.
