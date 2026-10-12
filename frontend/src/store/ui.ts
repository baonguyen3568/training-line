import { create } from "zustand";

export type Mode = "simple" | "advanced";
const KEY = "vlm.ui.mode";

// Per-viewer convenience only: storage may be unavailable (private window, blocked site data).
function readMode(): Mode {
  try {
    return localStorage.getItem(KEY) === "advanced" ? "advanced" : "simple";
  } catch {
    return "simple";
  }
}

interface UiState {
  /** Default Simple (user decision, ADR-009). */
  mode: Mode;
  setMode(mode: Mode): void;
}

export const useUi = create<UiState>((set) => ({
  mode: readMode(),
  setMode(mode) {
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* ignore */
    }
    set({ mode });
  },
}));
