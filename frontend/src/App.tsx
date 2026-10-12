import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Editor } from "./components/Editor";
import { Gallery } from "./components/Gallery";
import { Header, type Nav } from "./components/Header";
import { t } from "./i18n";
import { clearDraft, type Draft, loadDraft, saveDraft } from "./lib/draft";
import { useSystem } from "./store/system";
import { useWorkflow } from "./store/workflow";

// Run panels (and uPlot) load on demand to keep the editor bundle small.
const RunMonitor = lazy(() =>
  import("./components/RunMonitor").then((m) => ({ default: m.RunMonitor })),
);
const RunsList = lazy(() => import("./components/RunsList").then((m) => ({ default: m.RunsList })));

type View = { kind: "home" } | { kind: "runs" } | { kind: "run"; id: string };

export function App() {
  const workflow = useWorkflow((s) => s.workflow);
  const { loadCatalog, openTemplate, closeWorkflow, newWorkflow } = useWorkflow.getState();
  const [catalogError, setCatalogError] = useState(false);
  const [view, setView] = useState<View>({ kind: "home" });
  const worker = useSystem((s) => s.status?.worker);
  const [draft, setDraft] = useState<Draft | null>(() => loadDraft());

  // Autosave the open workflow (debounced 1s) so a reload never loses work (design review M1 K1).
  useEffect(() => {
    if (!workflow) return;
    const timer = setTimeout(() => saveDraft(workflow), 1000);
    return () => clearTimeout(timer);
  }, [workflow]);

  const load = useCallback(() => {
    setCatalogError(false);
    loadCatalog().catch(() => setCatalogError(true));
  }, [loadCatalog]);
  useEffect(load, [load]);

  const navigate = (to: Nav) => {
    if (to === "home") closeWorkflow();
    setView({ kind: to });
  };

  return (
    <div className="app">
      <Header onNavigate={navigate} active={view.kind === "home" ? "home" : "runs"} />
      {(import.meta.env.VITE_MOCK_API === "1" || worker === "fake") && (
        // Mock API: make it impossible to mistake the FakeWorker for real training (PM request).
        <p className="demo-banner" role="note">
          🧪 {t("app.demo_mode")}
        </p>
      )}
      {catalogError && (
        <p role="alert" className="banner">
          {t("app.catalog_load_failed")}{" "}
          <button type="button" className="link" onClick={load}>
            {t("app.retry")}
          </button>
        </p>
      )}
      <Suspense fallback={<div className="skeleton-grid" aria-busy="true" />}>
        {draft && !workflow && (
          <p className="banner draft" role="alert">
            {t("draft.restore_prompt", { time: new Date(draft.savedAt).toLocaleString() })}{" "}
            <button
              type="button"
              className="primary"
              onClick={() => {
                useWorkflow.getState().openWorkflow(draft.workflow);
                setDraft(null);
                setView({ kind: "home" });
              }}
            >
              {t("draft.restore")}
            </button>{" "}
            <button
              type="button"
              className="link"
              onClick={() => {
                clearDraft();
                setDraft(null);
              }}
            >
              {t("draft.discard")}
            </button>
          </p>
        )}
        {view.kind === "runs" && <RunsList onOpen={(id) => setView({ kind: "run", id })} />}
        {view.kind === "run" && <RunMonitor runId={view.id} />}
      </Suspense>
      {/* First screen is the template gallery, never a blank canvas (P010 §8). */}
      {view.kind === "home" &&
        (workflow ? (
          <Editor />
        ) : (
          <Gallery onOpen={(id) => void openTemplate(id)} onBlank={newWorkflow} />
        ))}
    </div>
  );
}
