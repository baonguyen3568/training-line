# UI copy — moved

All user-visible text now lives in **`src/vlm_pipeline/i18n/{vi,en}.yaml`** (single source, owner bao-design; ADR-009, 007 R9).
The former `*.draft.yaml` files here were merged into it on 2026-10-12 and deleted to avoid a second source.

## Rendering rules that are not visible in the YAML
- `custom_node.perm.isolation`: show **only** when the install plan has `sandbox_verified=true` (P003 D16 sandbox test passed).
- `custom_node.perm.network` / `typed_confirm_network`: v1.1 only; in v1 a network request is `blocked_reason.permission_unsupported`.
- `custom_node.shared_wf.*`: names taken from a workflow file are untrusted — plain text, truncate ~60 chars, in quotes,
  always followed by `shared_wf.name_from_file`; show `lookalike_warning` when backend returns `name_lookalike_of`.
  Never prefill the link input from the workflow file.
- `storage.dialog.approval_needed`: only when `plan.needs_approval=true` (approval_policy=threshold, shared dev machine).
- `storage.consequence.WILL_REDOWNLOAD.no_workflow`: when `consequence_params.workflow` is empty.
- `error.<CODE>.what.<variant>`: variant chosen by fixcode's catalog (`variants`/`variant_by`).
- `error.<CODE>.why` is OPTIONAL: only for a concrete runtime cause computed from params (e.g. MEM-003). Present → card shows "Vì sao"; absent → 3-part card. why ≤ 1 sentence.
- Error titles never contain placeholders. Title VI ≤ 10 tokens (`split()`), EN ≤ 8 words; what ≤ 2 sentences.
- Estimates: style by `Estimate.source` (P010 §6) — measured plain, formula "~", declared italic + ⓘ, unknown text.
- Permission `level`: info → text.secondary + neutral ✓ (never status.good); warning/critical → color.status.*.
