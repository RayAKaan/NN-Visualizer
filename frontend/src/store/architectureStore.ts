import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { LayerConfig, ValidationResult } from "../types/simulator";
import { simulatorApi } from "../hooks/useSimulatorApi";
import { useSimulatorStore } from "./simulatorStore";

interface ArchitectureState {
  layers: LayerConfig[];
  presetName: string | null;
  validationResult: ValidationResult | null;
  totalParams: number;
  flopsPerSample: number;
  addLayerOfType: (type: LayerConfig["type"]) => void;
  removeLayer: (index: number) => void;
  updateLayer: (index: number, updates: Partial<LayerConfig>) => void;
  loadPreset: (name: string) => void;
  setLayers: (layers: LayerConfig[], name?: string | null) => void;
  validate: () => Promise<void>;
  build: () => Promise<void>;
}

const defaultLayers: LayerConfig[] = [
  { type: "input", neurons: 2 },
  { type: "dense", neurons: 6, activation: "relu", init: "xavier" },
  { type: "dense", neurons: 4, activation: "relu", init: "xavier" },
  { type: "output", neurons: 1, activation: "sigmoid", init: "xavier" },
];

const presets: Record<string, LayerConfig[]> = {
  "Simple 2-4-1": [
    { type: "input", neurons: 2 },
    { type: "dense", neurons: 4, activation: "relu", init: "xavier" },
    { type: "output", neurons: 1, activation: "sigmoid", init: "xavier" },
  ],
  "Deep 2-8-8-4-1": [
    { type: "input", neurons: 2 },
    { type: "dense", neurons: 8, activation: "relu", init: "xavier" },
    { type: "dense", neurons: 8, activation: "relu", init: "xavier" },
    { type: "dense", neurons: 4, activation: "relu", init: "xavier" },
    { type: "output", neurons: 1, activation: "sigmoid", init: "xavier" },
  ],
  "Wide 2-16-1": [
    { type: "input", neurons: 2 },
    { type: "dense", neurons: 16, activation: "relu", init: "xavier" },
    { type: "output", neurons: 1, activation: "sigmoid", init: "xavier" },
  ],
  "Simple CNN": [
    { type: "input", neurons: 1, input_shape: [1, 28, 28] },
    { type: "conv2d", neurons: 8, filters: 8, kernel_size: 3, stride: 1, padding: "same", activation: "relu" },
    { type: "maxpool2d", neurons: 1, pool_size: 2, pool_stride: 2 },
    { type: "flatten", neurons: 1 },
    { type: "dense", neurons: 32, activation: "relu" },
    { type: "output", neurons: 10, activation: "softmax" },
  ],
  "LSTM Classifier": [
    { type: "input", neurons: 1, input_shape: [20, 1] },
    { type: "lstm", neurons: 32, hidden_size: 32, return_sequences: false },
    { type: "output", neurons: 3, activation: "softmax" },
  ],
};

const defaultsByType: Record<LayerConfig["type"], LayerConfig> = {
  input: { type: "input", neurons: 2 },
  dense: { type: "dense", neurons: 8, activation: "relu", init: "xavier" },
  output: { type: "output", neurons: 1, activation: "sigmoid", init: "xavier" },
  conv2d: { type: "conv2d", neurons: 8, filters: 8, kernel_size: 3, stride: 1, padding: "same", activation: "relu" },
  maxpool2d: { type: "maxpool2d", neurons: 1, pool_size: 2, pool_stride: 2 },
  avgpool2d: { type: "avgpool2d", neurons: 1, pool_size: 2, pool_stride: 2 },
  flatten: { type: "flatten", neurons: 1 },
  batchnorm: { type: "batchnorm", neurons: 1 },
  rnn: { type: "rnn", neurons: 16, hidden_size: 16, return_sequences: false },
  lstm: { type: "lstm", neurons: 32, hidden_size: 32, return_sequences: false },
  gru: { type: "gru", neurons: 32, hidden_size: 32, return_sequences: false },
  embedding: { type: "embedding", neurons: 16, vocab_size: 50, embedding_dim: 16 },
  attention: { type: "attention", neurons: 8, num_heads: 2 },
  residual: { type: "residual", neurons: 1 },
};

export const useArchitectureStore = create<ArchitectureState>()(
  devtools((set, get) => ({
    layers: defaultLayers,
    presetName: null,
    validationResult: null,
    totalParams: 0,
    flopsPerSample: 0,

    addLayerOfType(type) {
      const layers = [...get().layers];
      const insertIndex = Math.max(1, layers.length - 1);
      const defaults = defaultsByType[type] ?? defaultsByType.dense;
      layers.splice(insertIndex, 0, { ...defaults });
      set({ layers });
      void get().validate();
    },
    removeLayer(index) {
      const layers = get().layers.filter((_, i) => i !== index);
      set({ layers });
      void get().validate();
    },
    updateLayer(index, updates) {
      const layers = get().layers.map((l, i) => (i === index ? { ...l, ...updates } : l));
      set({ layers });
      void get().validate();
    },
    loadPreset(name) {
      const layers = presets[name] ?? defaultLayers;
      set({ layers, presetName: name });
      void get().validate();
    },
    setLayers(layers, name = null) {
      set({ layers, presetName: name });
      void get().validate();
    },
    async validate() {
      try {
        const result = await simulatorApi.validateArchitecture(get().layers);
        set({ validationResult: result, totalParams: result.total_params, flopsPerSample: result.flops_per_sample });
      } catch {
        set({ validationResult: null });
      }
    },
    async build() {
      const result = await simulatorApi.buildArchitecture(get().layers);
      useSimulatorStore.getState().setGraphId(result.graph_id);
    },
  })),
);
