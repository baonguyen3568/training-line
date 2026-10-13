import type { DownloadTarget } from "../components/DownloadDialog";
import { downloadKey } from "../store/downloads";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { downloadTargetOf } from "./downloadTarget";

/** What the open workflow still has to download (validate's NET-002 warns), one entry per asset. */
export function missingDownloads(): DownloadTarget[] {
  const { validation, specs } = useWorkflow.getState();
  const seen = new Map<string, DownloadTarget>();
  for (const issue of validation?.errors ?? []) {
    const target = downloadTargetOf(issue, specs);
    if (target) seen.set(downloadKey(target.kind, target.catalogId), target);
  }
  return [...seen.values()];
}

/**
 * Every "Chạy" goes through here (G-5 "Bấm Chạy khi thiếu"): with something not on this machine yet, the list
 * dialog opens and the run starts only after all of it downloaded; otherwise the run starts now.
 */
export function requestRun(): void {
  if (missingDownloads().length > 0) useUi.getState().setRunMissingOpen(true);
  else void useWorkflow.getState().startRun();
}
