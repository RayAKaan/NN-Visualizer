import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { AugmentationPreviewResponse } from "../types/simulator";

interface AugmentationState {
  loading: boolean;
  noise: number;
  nSamples: number;
  pipeline: Array<{ type: string; [key: string]: any }>;
  result: AugmentationPreviewResponse | null;
  setNoise: (value: number) => void;
  setNSamples: (value: number) => void;
  setPipeline: (pipeline: Array<{ type: string; [key: string]: any }>) => void;
  preview: (input: number[], inputShape?: number[] | null) => Promise<void>;
}

export const useAugmentationStore = create<AugmentationState>((set, get) => ({
  loading: false,
  noise: 0.1,
  nSamples: 8,
  result: null,
  pipeline: [],
  setNoise: (value) => set({ noise: value }),
  setNSamples: (value) => set({ nSamples: value }),
  setPipeline: (pipeline) => set({ pipeline }),
  async preview(input, inputShape) {
    const { nSamples, pipeline } = get();
    set({ loading: true });
    try {
      const res = await simulatorApi.augmentationPreview(input, inputShape ?? null, nSamples, pipeline);
      set({ result: res, loading: false });
    } catch {
      set({ loading: false });
    }
  },
}));
