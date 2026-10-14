import { Fragment, useEffect } from "react";
import { t } from "../i18n";
import { nodeTitle } from "../lib/nodeText";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { Icon } from "./Icon";

const upTip = () => `${t("canvas.group.up")} (Alt+↑)`;

/** True when Esc belongs to something else: an open popover, a dialog, or a field being typed in. */
function escTakenElsewhere(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return true;
  if (el?.isContentEditable) return true;
  if (document.querySelector("[role=dialog], [role=alertdialog]")) return true;
  // without the popover API (old browsers, jsdom) no popover can be open; its selector must not be trusted there
  if (!("showPopover" in HTMLElement.prototype)) return false;
  try {
    return !!document.querySelector("[popover]:popover-open");
  } catch {
    return false;
  }
}

/**
 * P4-4 §2: where the canvas is inside the workflow, after its name: "Workflow › Dạy model › Vòng học". Each part goes
 * up to that level; the last carries a lock (every group is view-only inside in 1.0, design 818e7c3). Alt+↑ goes up
 * one level, and Esc too when no popover, dialog or field has it.
 */
export function GroupBreadcrumb() {
  const path = useUi((st) => st.groupPath);
  const workflow = useWorkflow((st) => st.workflow);
  const specs = useWorkflow((st) => st.specs);

  useEffect(() => {
    if (path.length === 0) return;
    const onKey = (e: KeyboardEvent) => {
      const up = (e.altKey && e.key === "ArrowUp") || (e.key === "Escape" && !escTakenElsewhere(e));
      if (!up) return;
      e.preventDefault();
      useUi.getState().setGroupPath(path.slice(0, -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [path]);

  if (path.length === 0 || !workflow) return null;
  // the title of each entered group node, walking down the levels
  const crumbs: { id: string; title: string; user: boolean }[] = [];
  let nodes = workflow.nodes;
  for (const id of path) {
    const n = nodes.find((x) => x.id === id);
    const spec = n && specs.get(n.type);
    if (!n || !spec?.group) break;
    crumbs.push({ id, title: nodeTitle(spec, n.type), user: !!spec.group.user });
    nodes = spec.group.body.nodes;
  }
  const last = crumbs[crumbs.length - 1];
  const go = (depth: number) => useUi.getState().setGroupPath(path.slice(0, depth));

  return (
    <nav className="tb-crumbs" aria-label={t("canvas.group.breadcrumb_label")}>
      <ol>
        <li>
          <button
            type="button"
            className="link"
            // the level just above is where Alt+↑ goes: say so on it (design r1 a)
            data-tip={path.length === 1 ? upTip() : undefined}
            onClick={() => go(0)}
          >
            {t("canvas.group.root")}
          </button>
        </li>
        {crumbs.map((c, i) => (
          <Fragment key={c.id}>
            <li aria-hidden="true" className="tb-crumb-sep">
              <Icon name="chevron-right" size={12} />
            </li>
            <li>
              {i === crumbs.length - 1 ? (
                <span aria-current="page" className="tb-crumb-here">
                  {c.title}
                  {last && (
                    <span
                      className="si si-muted tb-crumb-lock"
                      data-tip={t(
                        last.user ? "canvas.group.readonly_user" : "canvas.group.readonly",
                      )}
                    >
                      <Icon
                        name="lock"
                        size={14}
                        label={t(
                          last.user ? "canvas.group.readonly_user" : "canvas.group.readonly",
                        )}
                      />
                    </span>
                  )}
                </span>
              ) : (
                <button
                  type="button"
                  className="link"
                  data-tip={i === crumbs.length - 2 ? upTip() : undefined}
                  onClick={() => go(i + 1)}
                >
                  {c.title}
                </button>
              )}
            </li>
          </Fragment>
        ))}
      </ol>
    </nav>
  );
}
