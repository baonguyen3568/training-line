import type { NodeSpec } from "../api/types";
import type { DownloadTarget } from "../components/DownloadDialog";

/**
 * The download a NET-002 asks for (catalog_id + kind + size from the server), from a validation issue or a step's
 * run error alike, with the model's label, licence, gated flag and publisher from the catalog (model.hf_pick
 * option_meta). `not_cached` and `not_prepared` are fixed by a download (the server prepares a bridged model after
 * it lands, 5a1d3ef); `not_pinned` is outside the vetted catalog, so it gets no button.
 */
export function downloadTargetOf(
  issue: { code: string; params?: Record<string, unknown> | undefined },
  specs: Map<string, NodeSpec>,
): DownloadTarget | undefined {
  const p = issue.params ?? {};
  if (issue.code !== "NET-002" || (p.reason !== "not_cached" && p.reason !== "not_prepared"))
    return undefined;
  const catalogId = typeof p.catalog_id === "string" ? p.catalog_id : undefined;
  const kind = p.kind === "model" || p.kind === "dataset" ? p.kind : undefined;
  if (!catalogId || !kind) return undefined;
  const meta =
    kind === "model"
      ? specs.get("model.hf_pick")?.params_schema.properties.model?.["x-ui"]?.option_meta?.[
          catalogId
        ]
      : undefined;
  const org = typeof p.repo === "string" && p.repo.includes("/") ? p.repo.split("/")[0] : undefined;
  const publisher = meta?.vendor_name ?? org;
  // datasets have no option_meta: the server's catalog name / licence in the issue, when it sends them
  const str = (v: unknown) => (typeof v === "string" && v ? v : undefined);
  const license = meta?.license ?? str(p.license);
  return {
    kind,
    catalogId,
    name: meta?.label ?? str(p.name) ?? catalogId,
    ...(typeof p.size_gb === "number" ? { sizeGb: p.size_gb } : {}),
    // the Hugging Face repo, for the gated flow's "open the model page" link: a catalog repo id shape only
    ...(typeof p.repo === "string" && /^[\w.-]+\/[\w.-]+$/.test(p.repo) ? { repo: p.repo } : {}),
    ...(license ? { license } : {}),
    ...(meta?.license_class ? { licenseClass: meta.license_class } : {}),
    ...(meta?.obligations?.length ? { obligations: meta.obligations } : {}),
    ...(meta?.gated ? { gated: true } : {}),
    // a proper noun, never translated nor shown as the slug (design G-5 a): the catalog name, else the repo's org
    ...(publisher ? { publisher } : {}),
    ...(typeof p.est_min === "number" ? { estMin: p.est_min } : {}),
    // already on disk, only the preparation step is left: nothing big to fetch (design d9e43a9)
    ...(p.reason === "not_prepared" ? { prepare: true } : {}),
  };
}
