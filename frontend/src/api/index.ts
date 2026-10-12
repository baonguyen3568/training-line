import { type Api, httpApi } from "./client";

// Dev/test only: the mock is loaded lazily so `vite build` (VITE_MOCK_API unset) tree-shakes it out.
function lazyMock(): Api {
  const load = () => import("../mocks/api").then((m) => m.mockApi);
  return {
    nodeTypes: (locale) => load().then((m) => m.nodeTypes(locale)),
    templates: () => load().then((m) => m.templates()),
    template: (id) => load().then((m) => m.template(id)),
    validate: (wf, locale) => load().then((m) => m.validate(wf, locale)),
    system: () => load().then((m) => m.system()),
    runs: () => load().then((m) => m.runs()),
    run: (id) => load().then((m) => m.run(id)),
    runMetrics: (id, q) => load().then((m) => m.runMetrics(id, q)),
    startRun: (wf) => load().then((m) => m.startRun(wf)),
    cancelRun: (id) => load().then((m) => m.cancelRun(id)),
    watchRun: (id, onEvent) => {
      let stop: (() => void) | null = null;
      let cancelled = false;
      void load().then((m) => {
        if (!cancelled) stop = m.watchRun(id, onEvent);
      });
      return () => {
        cancelled = true;
        stop?.();
      };
    },
  };
}

export const api: Api = import.meta.env.VITE_MOCK_API === "1" ? lazyMock() : httpApi;
export * from "./types";
