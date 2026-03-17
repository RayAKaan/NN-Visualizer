import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { GenerativeSampleResponse } from "../types/simulator";

interface GenerativeState {
  loading: boolean;
  mode: "vae" | "gan" | "ae";
  nSamples: number;
  size: number;
  result: GenerativeSampleResponse | null;
  epochs: number;
  metrics: Record<string, any> | null;
  setMode: (value: "vae" | "gan" | "ae") => void;
  setNSamples: (value: number) => void;
  setSize: (value: number) => void;
  setEpochs: (value: number) => void;
  train: (datasetId: string) => Promise<void>;
  sample: () => Promise<void>;
}

export const useGenerativeStore = create<GenerativeState>((set, get) => ({
  loading: false,
  mode: "vae",
  nSamples: 8,
  size: 28,
  result: null,
  epochs: 5,
  metrics: null,
  setMode: (value) => set({ mode: value }),
  setNSamples: (value) => set({ nSamples: value }),
  setSize: (value) => set({ size: value }),
  setEpochs: (value) => set({ epochs: value }),
  async train(datasetId) {
    const { mode, epochs } = get();
    set({ loading: true });
    try {
      const res = await simulatorApi.generativeTrain(datasetId, mode, epochs);
      set({ metrics: res.metrics ?? null, loading: false });
    } catch {
      set({ loading: false });
    }
  },
  async sample() {
    const { mode, nSamples, size } = get();
    set({ loading: true });
    try {
      const res = await simulatorApi.generativeSample(mode, nSamples, size);
      set({ result: res, loading: false });
    } catch {
      set({ loading: false });
    }
  },
}));
