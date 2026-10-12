import { create } from "zustand";
import type { SystemStatus } from "../api/types";

/** Latest /api/system snapshot (polled by the header); used for dynamic RAM-fit badges. */
export const useSystem = create<{ status: SystemStatus | null; set(s: SystemStatus | null): void }>(
  (set) => ({
    status: null,
    set: (status) => set({ status }),
  }),
);
