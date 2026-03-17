import { create } from "zustand";

interface FlowStore {
  isRibbonVisible: boolean;
  toggleRibbon: () => void;
  setRibbonVisible: (value: boolean) => void;
}

export const useFlowStore = create<FlowStore>((set) => ({
  isRibbonVisible: true,
  toggleRibbon: () => set((s) => ({ isRibbonVisible: !s.isRibbonVisible })),
  setRibbonVisible: (value) => set({ isRibbonVisible: value }),
}));
