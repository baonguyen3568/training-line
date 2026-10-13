// SPDX-ish licence ids of the catalog and reference pipelines → display names (unknown ids shown as written,
// never invented). Shared by the pipeline gallery and the download dialog.
export const LICENSE_NAME: Record<string, string> = {
  "apache-2.0": "Apache-2.0",
  mit: "MIT",
  "cc-by-4.0": "CC-BY-4.0",
  "nvidia-open-model-license": "NVIDIA Open Model License",
  "openmdw-1.1": "OpenMDW-1.1",
};
export const licenseName = (id: string) => LICENSE_NAME[id] ?? id;
