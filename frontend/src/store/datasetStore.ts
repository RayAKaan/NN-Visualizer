import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { DatasetPoint, DatasetResponse } from "../types/simulator";
import { simulatorApi } from "../hooks/useSimulatorApi";

interface DatasetState {
  datasetId: string | null;
  datasetType: string;
  datasetCategory: "synthetic" | "image" | "sequence" | "custom";
  standardName: string;
  sequenceType: string;
  seqLength: number;
  nFeatures: number;
  nClasses: number;
  vocabSize: number;
  nSamples: number;
  noise: number;
  trainSplit: number;
  trainData: DatasetPoint[];
  testData: DatasetPoint[];
  standardData: any | null;
  sequenceData: any | null;
  customPoints: DatasetPoint[];
  isCustomMode: boolean;
  stats: DatasetResponse["stats"] | null;
  generate: () => Promise<void>;
  loadStandard: () => Promise<void>;
  generateSequence: () => Promise<void>;
  addCustomPoint: (x: number[], y: number[]) => void;
  removeCustomPoint: (index: number) => void;
  clearCustom: () => void;
  submitCustom: () => Promise<void>;
}

export const useDatasetStore = create<DatasetState>()(
  devtools((set, get) => ({
  datasetType: "spiral",
  datasetCategory: "synthetic",
  standardName: "mnist",
  sequenceType: "sine_wave",
  seqLength: 20,
  nFeatures: 1,
  nClasses: 3,
  vocabSize: 50,
  nSamples: 200,
  noise: 0.15,
  trainSplit: 0.8,
  datasetId: null,
  trainData: [],
  testData: [],
  standardData: null,
  sequenceData: null,
    customPoints: [],
    isCustomMode: false,
    stats: null,

    async generate() {
      const { datasetType, nSamples, noise, trainSplit } = get();
      const res = await simulatorApi.datasetGenerate({
        type: datasetType,
        n_samples: nSamples,
        noise,
        train_split: trainSplit,
      });
      set({ trainData: res.train, testData: res.test, stats: res.stats, datasetId: res.dataset_id });
    },
    async loadStandard() {
      const { standardName, nSamples, trainSplit } = get();
      const res = await simulatorApi.datasetLoadStandard({
        name: standardName,
        n_samples: nSamples,
        train_split: trainSplit,
      });
      set({ standardData: res, datasetId: res.dataset_id });
    },
    async generateSequence() {
      const { sequenceType, nSamples, seqLength, nFeatures, nClasses, vocabSize, noise, trainSplit } = get();
      const res = await simulatorApi.datasetGenerateSequence({
        type: sequenceType,
        n_samples: nSamples,
        seq_length: seqLength,
        n_features: nFeatures,
        vocab_size: vocabSize,
        n_classes: nClasses,
        noise,
        train_split: trainSplit,
      });
      set({ sequenceData: res, datasetId: res.dataset_id });
    },
    addCustomPoint(x, y) {
      set((s) => ({ customPoints: [...s.customPoints, { x, y }] }));
    },
    removeCustomPoint(index) {
      set((s) => ({ customPoints: s.customPoints.filter((_, i) => i !== index) }));
    },
    clearCustom() {
      set({ customPoints: [] });
    },
    async submitCustom() {
      const { customPoints, trainSplit } = get();
      if (customPoints.length === 0) return;
      const res = await simulatorApi.datasetCustom({ points: customPoints, train_split: trainSplit });
      set({ trainData: res.train, testData: res.test, stats: res.stats, datasetId: res.dataset_id });
    },
  })),
);
