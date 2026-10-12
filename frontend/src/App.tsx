import { useEffect, useState } from "react";
import { Editor } from "./components/Editor";
import { Gallery } from "./components/Gallery";
import { Header } from "./components/Header";
import { t } from "./i18n";
import { useWorkflow } from "./store/workflow";

export function App() {
  const workflow = useWorkflow((s) => s.workflow);
  const { loadCatalog, openTemplate, closeWorkflow } = useWorkflow.getState();
  const [catalogError, setCatalogError] = useState(false);

  useEffect(() => {
    loadCatalog().catch(() => setCatalogError(true));
  }, [loadCatalog]);

  return (
    <div className="app">
      <Header onHome={closeWorkflow} />
      {catalogError && (
        <p role="alert" className="banner">
          {t("error.SYS-001.title")}
        </p>
      )}
      {/* First screen is the template gallery, never a blank canvas (P010 §8). */}
      {workflow ? <Editor /> : <Gallery onOpen={(id) => void openTemplate(id)} />}
    </div>
  );
}
