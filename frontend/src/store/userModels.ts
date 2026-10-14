import { create } from "zustand";
import { api } from "../api";
import type { UserModel } from "../api/types";

/** A repo + pinned revision to put in a step's "Từ Hugging Face" panel (MODEL-006 "Thêm model này"), not checked yet. */
export interface HubPrefill {
  nodeId: string;
  repo: string;
  revision: string;
}

interface UserModelsState {
  /** GET /api/user-assets/models: the models this machine has a record of. */
  models: UserModel[];
  load(): Promise<void>;
  prefill: HubPrefill | null;
  /** Open the step's hub tab with this repo and revision typed in; the user still presses "Kiểm tra". */
  openAdd(prefill: HubPrefill): void;
  /** The prefill for this step, once: the panel reads it and it is gone. */
  takePrefill(nodeId: string): HubPrefill | null;
}

export const useUserModels = create<UserModelsState>((set, get) => ({
  models: [],
  async load() {
    try {
      const { models } = await api.listUserModels();
      set({ models });
    } catch {
      /* the list is a shortcut; pasting the repo again still works */
    }
  },
  prefill: null,
  openAdd(prefill) {
    set({ prefill });
  },
  takePrefill(nodeId) {
    const p = get().prefill;
    if (!p || p.nodeId !== nodeId) return null;
    set({ prefill: null });
    return p;
  },
}));
