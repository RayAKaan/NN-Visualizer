import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { EmbeddingResponse } from "../types/simulator";

interface EmbeddingState {
  loading: boolean;
  error: string | null;
  layerIndex: number;
  nSamples: number;
  method: "pca" | "tsne" | "umap";
  result: EmbeddingResponse | null;
  setLayerIndex: (value: number) => void;
  setNSamples: (value: number) => void;
  setMethod: (value: "pca" | "tsne" | "umap") => void;
  compute: (graphId: string, datasetId: string) => Promise<void>;
  clear: () => void;
}

export const useEmbeddingStore = create<EmbeddingState>((set, get) => ({
  loading: false,
  error: null,
  layerIndex: 0,
  nSamples: 200,
  method: "pca",
  result: null,
  setLayerIndex: (value) => set({ layerIndex: value }),
  setNSamples: (value) => set({ nSamples: value }),
  setMethod: (value) => set({ method: value }),
  async compute(graphId, datasetId) {
    const { layerIndex, nSamples, method } = get();
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.embeddingsCompute(graphId, datasetId, layerIndex, nSamples, method);
      set({ result: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Failed to compute embeddings." });
    }
  },
  clear() {
    set({ result: null, error: null });
  },
}));
