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
  };
}

export const api: Api = import.meta.env.VITE_MOCK_API === "1" ? lazyMock() : httpApi;
export * from "./types";
