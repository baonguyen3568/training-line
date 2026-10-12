import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

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
  useUi.setState({ mode: "simple" });
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
