import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

// Shared host: with a GPU job next door (load 9-14) a cold first render can take > 1 s, so findBy*/waitFor
// timed out in full runs while every file passed alone (run-monitor, storage-panel, settings-panel, 2026-10-13).
configure({ asyncUtilTimeout: 5000 });

// Global zustand stores outlive a test: reset them so results never depend on test order.
// Imported lazily so test files can stub VITE_MOCK_API before src/api is first evaluated.
let initialWorkflow: unknown;
afterEach(async () => {
  cleanup();
  const [{ useUi }, { useWorkflow }] = await Promise.all([
    import("../src/store/ui"),
    import("../src/store/workflow"),
  ]);
  initialWorkflow ??= useWorkflow.getInitialState();
  useUi.setState({ mode: "simple", skin: "d4" }); // the 1.0 default, whatever a test pinned
  useWorkflow.setState(initialWorkflow as ReturnType<typeof useWorkflow.getState>, true);
});

// jsdom lacks ResizeObserver/DOMMatrix used by React Flow.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

// jsdom lacks matchMedia (uPlot calls it at import time).
if (typeof window.matchMedia !== "function") {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}
