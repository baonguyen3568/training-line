import { create } from "zustand";

export type Mode = "simple" | "advanced";
/** Release 1.0: the D4 shell is the default; `?ui=classic` (remembered) keeps the classic layout, `?ui=d4` returns. */
export type Skin = "classic" | "d4";
const KEY = "vlm.ui.mode";
const SKIN_KEY = "vlm.ui.skin";

function readSkin(): Skin {
  try {
    const q = new URLSearchParams(location.search).get("ui");
    if (q === "d4" || q === "classic" || q === "legacy") {
      const skin: Skin = q === "d4" ? "d4" : "classic"; // "legacy": alias in design's plan, same as classic
      localStorage.setItem(SKIN_KEY, skin);
      return skin;
    }
    return localStorage.getItem(SKIN_KEY) === "classic" ? "classic" : "d4";
  } catch {
    return "d4";
  }
}

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
  skin: Skin;
  /** D4 bottom dock: collapsed to the tab row by default; opens on a tab click or when pre-flight finds errors. */
  dockOpen: boolean;
  setDockOpen(open: boolean): void;
  libraryOpen: boolean;
  setLibraryOpen(open: boolean): void;
  /** The wire being dragged (port it started from), so ports it cannot reach are dimmed (design). */
  wireDrag: WireDrag | null;
  setWireDrag(drag: WireDrag | null): void;
}

export interface WireDrag {
  nodeId: string;
  side: "source" | "target";
  type: string;
}

export const useUi = create<UiState>((set) => ({
  mode: readMode(),
  skin: readSkin(),
  dockOpen: false,
  setDockOpen(dockOpen) {
    set({ dockOpen });
  },
  libraryOpen: false,
  setLibraryOpen(libraryOpen) {
    set({ libraryOpen });
  },
  wireDrag: null,
  setWireDrag(wireDrag) {
    set({ wireDrag });
  },
  setMode(mode) {
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* ignore */
    }
    set({ mode });
  },
}));
