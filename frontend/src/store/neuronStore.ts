import { create } from "zustand";
import { apiClient } from "../api/client";
import type { Architecture, Dataset } from "../types/pipeline";
import type { NeuronIdentity } from "../types/neuron";

interface NeuronStore {
  isOpen: boolean;
  selected: { stageId: string; neuronIndex: number } | null;
  neuron: NeuronIdentity | null;
  isLoading: boolean;
  error: string | null;
  openNeuron: (
    stageId: string,
    neuronIndex: number,
    architecture: Architecture,
    dataset: Dataset,
    pixels: Float32Array,
  ) => Promise<void>;
  closeNeuron: () => void;
}

export const useNeuronStore = create<NeuronStore>((set) => ({
  isOpen: false,
  selected: null,
  neuron: null,
  isLoading: false,
  error: null,

  async openNeuron(stageId, neuronIndex, architecture, dataset, pixels) {
    set({ isOpen: true, selected: { stageId, neuronIndex }, isLoading: true, error: null });
    try {
      const { data } = await apiClient.post("/api/lab/neuron-biography", {
        stageId,
        neuronIndex,
        architecture,
        dataset,
        pixels: Array.from(pixels),
      });
      set({ neuron: data as NeuronIdentity, isLoading: false });
    } catch {
      set({ error: "Failed to load neuron", isLoading: false });
    }
  },

  closeNeuron() {
    set({ isOpen: false, selected: null, neuron: null, isLoading: false, error: null });
  },
}));
