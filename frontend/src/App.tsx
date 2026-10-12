import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Editor } from "./components/Editor";
import { Gallery } from "./components/Gallery";
import { Header, type Nav } from "./components/Header";
import { t } from "./i18n";
import { useWorkflow } from "./store/workflow";

// Run panels (and uPlot) load on demand to keep the editor bundle small.
const RunMonitor = lazy(() =>
  import("./components/RunMonitor").then((m) => ({ default: m.RunMonitor })),
);
const RunsList = lazy(() => import("./components/RunsList").then((m) => ({ default: m.RunsList })));

type View = { kind: "home" } | { kind: "runs" } | { kind: "run"; id: string };

export function App() {
  const workflow = useWorkflow((s) => s.workflow);
  const { loadCatalog, openTemplate, closeWorkflow } = useWorkflow.getState();
  const [catalogError, setCatalogError] = useState(false);
  const [view, setView] = useState<View>({ kind: "home" });

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
      {catalogError && (
        <p role="alert" className="banner">
          {t("app.catalog_load_failed")}{" "}
          <button type="button" className="link" onClick={load}>
            {t("app.retry")}
          </button>
        </p>
      )}
      <Suspense fallback={<div className="skeleton-grid" aria-busy="true" />}>
        {view.kind === "runs" && <RunsList onOpen={(id) => setView({ kind: "run", id })} />}
        {view.kind === "run" && <RunMonitor runId={view.id} />}
      </Suspense>
      {/* First screen is the template gallery, never a blank canvas (P010 §8). */}
      {view.kind === "home" &&
        (workflow ? <Editor /> : <Gallery onOpen={(id) => void openTemplate(id)} />)}
    </div>
  );
}
