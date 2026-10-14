import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { attrValue } from "./dom";

/**
 * "Mở bước bị lỗi" / "Chỉ cho tôi" for a step inside a group (P4-4 §3): enter the right level, select the step, then
 * focus the param (or port) the issue names once the inspector shows it.
 */
export function openInnerStep(
  trail: { groupPath: string[]; stepId: string },
  focus?: { param?: string | undefined; port?: string | undefined },
): void {
  useUi.getState().setGroupPath(trail.groupPath);
  useWorkflow.getState().select(trail.stepId);
  const target = focus?.param ? `param-${focus.param}` : focus?.port ? `recv-${focus.port}` : null;
  if (!target) return;
  let frames = 10; // the level and the inspector render on the next frames
  // only the inner step's own inspector: the group's inspector (still on screen this frame) has the same ids
  const tryFocus = () => {
    const el = document.querySelector<HTMLElement>(
      `.inspector.inner-step [id="${attrValue(target)}"]`,
    );
    if (el) el.focus();
    else if (frames-- > 0) requestAnimationFrame(tryFocus);
  };
  requestAnimationFrame(tryFocus);
}
