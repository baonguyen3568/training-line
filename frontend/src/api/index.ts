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
    storagePlan: (locale) => load().then((m) => m.storagePlan(locale)),
    storageApply: (body, locale) => load().then((m) => m.storageApply(body, locale)),
    bugReport: (runId, q, locale) => load().then((m) => m.bugReport(runId, q, locale)),
    nodeResetPlan: (node, locale) => load().then((m) => m.nodeResetPlan(node, locale)),
    advancedOptions: (typeId, model) => load().then((m) => m.advancedOptions(typeId, model)),
    health: () => load().then((m) => m.health()),
    lan: () => load().then((m) => m.lan()),
    session: () => load().then((m) => m.session()),
    signOut: () => load().then((m) => m.signOut()),
    revokeAllSessions: () => load().then((m) => m.revokeAllSessions()),
    unlockLan: () => load().then((m) => m.unlockLan()),
    addUserModel: (body) => load().then((m) => m.addUserModel(body)),
    userModelResolve: (id) => load().then((m) => m.userModelResolve(id)),
    confirmUserModelLicense: (id) => load().then((m) => m.confirmUserModelLicense(id)),
    listUserModels: () => load().then((m) => m.listUserModels()),
    inspectDataset: (body) => load().then((m) => m.inspectDataset(body)),
    datasetInspection: (id) => load().then((m) => m.datasetInspection(id)),
    startDownload: (body) => load().then((m) => m.startDownload(body)),
    watchDownload: (id, onEvent) => {
      let stop = () => {};
      let stopped = false;
      void load().then((m) => {
        if (!stopped) stop = m.watchDownload(id, onEvent);
      });
      return () => {
        stopped = true;
        stop();
      };
    },
    nodeReset: (body, locale) => load().then((m) => m.nodeReset(body, locale)),
    upload: (file, locale) => load().then((m) => m.upload(file, locale)),
    nodeResult: (runId, nodeId) => load().then((m) => m.nodeResult(runId, nodeId)),
    settings: () => load().then((m) => m.settings()),
    putSecret: (f, v) => load().then((m) => m.putSecret(f, v)),
    deleteSecret: (f) => load().then((m) => m.deleteSecret(f)),
    run: (id) => load().then((m) => m.run(id)),
    runMetrics: (id, q) => load().then((m) => m.runMetrics(id, q)),
    startRun: (wf, trial) => load().then((m) => m.startRun(wf, trial)),
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
