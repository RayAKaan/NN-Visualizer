import { create } from "zustand";

interface SpatialStore {
  isAttentionOpen: boolean;
  toggleAttention: () => void;
  closeAttention: () => void;
}

export const useSpatialStore = create<SpatialStore>((set) => ({
  isAttentionOpen: false,
  toggleAttention: () => set((s) => ({ isAttentionOpen: !s.isAttentionOpen })),
  closeAttention: () => set({ isAttentionOpen: false }),
}));
