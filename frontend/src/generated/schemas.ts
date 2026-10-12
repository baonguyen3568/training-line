// GENERATED — do not edit. Source: schemas/*.schema.json (backend). Regenerate: node scripts/gen-schema-types.mjs
/* biome-ignore-all lint: generated file */

/** Response of ``GET /api/runs/{id}/metrics?after=&limit=[&downsample_to=]`` (P009 §6, frontend contract). */
export type MetricsPage = {
  downsampled: boolean;
  has_more: boolean;
  next_cursor: string;
  points: ({
    [extra: string]: number | string | string[] | null | undefined;
  })[];
};

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" | "evicted" | "discarded";

/** ``run.json``. Unknown keys are kept on read (forward compatible) but never written by this version. */
export type RunManifest = {
  base_model?: string | null;
  base_model_revision?: string | null;
  config_hash?: string | null;
  config_path?: string | null;
  container_digest?: string | null;
  container_image?: string | null;
  dataset?: string | null;
  dataset_revision?: string | null;
  ended_at?: number | null;
  error_code?: string | null;
  exit_code?: number | null;
  git_commit?: string | null;
  git_dirty?: boolean | null;
  max_steps?: number | null;
  name: string;
  oom_killed?: boolean | null;
  parent_run_id?: string | null;
  peak_mem_gb?: number | null;
  pid?: number | null;
  run_id: string;
  schema_version?: number;
  seed?: number | null;
  stage: "data" | "train" | "quantize" | "eval" | "bench" | "serve" | "export";
  started_at?: number;
  status?: RunStatus;
  tags?: string[];
  task: string;
  [extra: string]: unknown;
};
