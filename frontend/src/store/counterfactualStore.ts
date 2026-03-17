import { create } from "zustand";
import { apiClient } from "../api/client";
import type { Architecture, Dataset } from "../types/pipeline";
import type { CounterfactualExperiment, SensitivityMapData } from "../types/counterfactual";

interface CounterfactualStore {
  isOpen: boolean;
  isBusy: boolean;
  sensitivityMap: SensitivityMapData | null;
  experiments: CounterfactualExperiment[];
  openExplorer: () => void;
  closeExplorer: () => void;
  computeSensitivity: (pixels: Float32Array, architecture: Architecture, dataset: Dataset) => Promise<void>;
  runCounterfactual: (
    originalPixels: Float32Array,
    modifiedPixels: Float32Array,
    architecture: Architecture,
    dataset: Dataset,
  ) => Promise<void>;
  runMinimalFlip: (pixels: Float32Array, architecture: Architecture, dataset: Dataset) => Promise<void>;
}

function toMask(a: Float32Array, b: Float32Array): Float32Array {
  const len = Math.min(a.length, b.length);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i += 1) out[i] = Math.abs(a[i] - b[i]);
  return out;
}

export const useCounterfactualStore = create<CounterfactualStore>((set) => ({
  isOpen: false,
  isBusy: false,
  sensitivityMap: null,
  experiments: [],

  openExplorer: () => set({ isOpen: true }),
  closeExplorer: () => set({ isOpen: false }),

  async computeSensitivity(pixels, architecture, dataset) {
    set({ isBusy: true });
    try {
      const { data } = await apiClient.post("/api/lab/sensitivity-map", {
        pixels: Array.from(pixels),
        architecture,
        dataset,
      });
      set({
        sensitivityMap: {
          perPixelSensitivity: new Float32Array(data.perPixelSensitivity ?? []),
          topSensitivePixels: data.topSensitivePixels ?? [],
          overallSensitivity: data.overallSensitivity ?? 0,
        },
      });
    } finally {
      set({ isBusy: false });
    }
  },

  async runCounterfactual(originalPixels, modifiedPixels, architecture, dataset) {
    set({ isBusy: true });
    try {
      const { data } = await apiClient.post("/api/lab/counterfactual", {
        originalPixels: Array.from(originalPixels),
        modifiedPixels: Array.from(modifiedPixels),
        architecture,
        dataset,
      });
      const exp: CounterfactualExperiment = {
        id: `exp-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        timestamp: Date.now(),
        originalInput: new Float32Array(originalPixels),
        modifiedInput: new Float32Array(modifiedPixels),
        perturbationMask: toMask(originalPixels, modifiedPixels),
        originalPrediction: data.originalPrediction,
        modifiedPrediction: data.modifiedPrediction,
        predictionFlipped: data.predictionFlipped,
        confidenceChange: data.confidenceChange,
        perturbationMagnitude: data.perturbationMagnitude,
        affectedPixelCount: data.affectedPixelCount,
      };
      set((s) => ({ experiments: [exp, ...s.experiments].slice(0, 20) }));
    } finally {
      set({ isBusy: false });
    }
  },

  async runMinimalFlip(pixels, architecture, dataset) {
    set({ isBusy: true });
    try {
      const { data } = await apiClient.post("/api/lab/minimal-flip", {
        pixels: Array.from(pixels),
        architecture,
        dataset,
      });
      if (!data?.found || !Array.isArray(data.modifiedPixels)) return;
      const modified = new Float32Array(data.modifiedPixels);
      const exp: CounterfactualExperiment = {
        id: `flip-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        timestamp: Date.now(),
        originalInput: new Float32Array(pixels),
        modifiedInput: modified,
        perturbationMask: toMask(pixels, modified),
        originalPrediction: {
          label: "original",
          confidence: 0,
          probs: [],
        },
        modifiedPrediction: data.newPrediction,
        predictionFlipped: true,
        confidenceChange: data.epsilon ?? 0,
        perturbationMagnitude: data.epsilon ?? 0,
        affectedPixelCount: data.affectedPixelCount ?? 0,
      };
      set((s) => ({ experiments: [exp, ...s.experiments].slice(0, 20) }));
    } finally {
      set({ isBusy: false });
    }
  },
}));
