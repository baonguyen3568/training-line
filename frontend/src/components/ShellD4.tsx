import { type ReactNode, useEffect, useRef } from "react";
import { t } from "../i18n";
import { nodeTitle } from "../lib/nodeText";
import { stepNumbers } from "../lib/topo";
import { useUi } from "../store/ui";
import { useWorkflow } from "../store/workflow";
import { Icon } from "./Icon";
import { IconButton } from "./IconButton";

/** D4 left icon strip (48px): library flyout toggle; storage pinned at the bottom (DESIGN.md Layout). */
export function IconStrip() {
  const open = useUi((s) => s.libraryOpen);
  const setOpen = useUi((s) => s.setLibraryOpen);
  return (
    <nav className="strip" aria-label={t("d4.tools")}>
      <IconButton
        id="strip-library"
        icon="boxes"
        label={t("d4.library")}
        shortcut="/"
        tipSide="right"
        aria-pressed={open}
        aria-controls="library-flyout"
        onClick={() => setOpen(!open)}
      />
      <span className="strip-grow" />
      <IconButton
        icon="hard-drive"
        label={t("app.nav_storage")}
        tipSide="right"
        onClick={() => window.dispatchEvent(new CustomEvent("vlm:navigate", { detail: "storage" }))}
      />
    </nav>
  );
}

/**
 * Library flyout (304px) over the canvas, not modal: opens on "/" or the strip icon, focuses its search box,
 * Esc closes and returns focus to the strip icon; it closes itself once a node is added (design M2-6 #3).
 */
export function LibraryFlyout({ children }: { children: ReactNode }) {
  const open = useUi((s) => s.libraryOpen);
  const setOpen = useUi((s) => s.setLibraryOpen);
  const nodeCount = useWorkflow((s) => s.workflow?.nodes.length ?? 0);
  const count = useRef(nodeCount);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing =
        e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
      if (e.key === "/" && !typing) {
        e.preventDefault();
        setOpen(true);
      } else if (e.key === "Escape" && useUi.getState().libraryOpen) {
        setOpen(false);
        document.getElementById("strip-library")?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setOpen]);

  useEffect(() => {
    if (open) panel.current?.querySelector<HTMLInputElement>("input[type=search]")?.focus();
  }, [open]);

  useEffect(() => {
    if (nodeCount > count.current && useUi.getState().libraryOpen) setOpen(false);
    count.current = nodeCount;
  }, [nodeCount, setOpen]);

  return (
    <div id="library-flyout" ref={panel} className="flyout" hidden={!open}>
      <div className="flyout-head">
        <strong>{t("d4.library")}</strong>
        <IconButton icon="x" label={t("d4.close")} shortcut="Esc" onClick={() => setOpen(false)} />
      </div>
      {children}
    </div>
  );
}

/**
 * Bottom dock: collapsed to its 32px tab row by default (the canvas gets the space); opens to 176px on a tab
 * click, or by itself when pre-flight finds blocking errors (design M2-6 #4).
 */
export function Dock({ children, errorCount }: { children: ReactNode; errorCount: number }) {
  const open = useUi((s) => s.dockOpen);
  const setOpen = useUi((s) => s.setDockOpen);
  const seen = useRef(0);
  useEffect(() => {
    if (errorCount > 0 && seen.current === 0) setOpen(true);
    seen.current = errorCount;
  }, [errorCount, setOpen]);
  return (
    <section className={`dock${open ? " open" : ""}`} aria-label={t("d4.dock")}>
      <div className="dock-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          className="dock-tab"
          aria-selected={open}
          aria-label={t("preflight.title")}
          data-tip={t("preflight.title")}
          data-tip-side="up"
          onClick={() => setOpen(!open)}
        >
          <Icon name="list-checks" />
          <span className={`count${errorCount ? " bad" : ""}`}>{errorCount}</span>
        </button>
        <span className="strip-grow" />
        <IconButton
          icon={open ? "chevron-down" : "chevrons-down-up"}
          label={open ? t("d4.dock_collapse") : t("d4.dock_expand")}
          tipSide="up"
          onClick={() => setOpen(!open)}
        />
      </div>
      <div className="dock-body" hidden={!open}>
        {children}
      </div>
    </section>
  );
}

/**
 * D4 status bar (26px, full width): the live run line ("Đang chạy bước 4/5 · Fine-tune toàn bộ") with a mini
 * progress bar, so progress is visible while the dock stays collapsed (design D4 r2 #2). Idle: nothing to say.
 */
export function StatusBar() {
  const { workflow, run, specs } = useWorkflow();
  if (!workflow || !run) return <footer className="sb" aria-hidden="true" />;
  const states = Object.entries(run.nodes);
  const done = states.filter(([, s]) =>
    ["completed", "cached", "skipped"].includes(s.status),
  ).length;
  const current = states.find(([, s]) => s.status === "running");
  const node = current && workflow.nodes.find((n) => n.id === current[0]);
  const steps = stepNumbers(workflow);
  const text = run.done
    ? t(`run_monitor.status.${run.status === "completed" ? "completed" : (run.status ?? "failed")}`)
    : node
      ? t("run.progress", {
          current_step: steps.get(node.id) ?? done + 1,
          total_count: states.length,
          node: nodeTitle(specs.get(node.type), node.type),
        })
      : t("node_state.queued");
  const pct = states.length ? Math.round((done / states.length) * 100) : 0;
  return (
    <footer className="sb" role="status" aria-live="polite">
      <span className={`si ${run.done ? "si-completed" : "si-running"}`}>
        <Icon name={run.done ? "circle-check-big" : "loader-circle"} size={14} />
      </span>
      <span>{text}</span>
      {!run.done && (
        <span
          className="sb-bar"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-label={text}
        >
          <span style={{ width: `${pct}%` }} />
        </span>
      )}
    </footer>
  );
}
