import { create } from "zustand";
import type { TrainingConfig, TrainingMetrics } from "../types/simulator";

interface TrainingSimState {
  config: TrainingConfig;
  isTraining: boolean;
  isPaused: boolean;
  currentEpoch: number;
  totalEpochs: number;
  currentBatch: number;
  totalBatches: number;
  metricsHistory: TrainingMetrics[];
  warnings: any[];
  updateConfig: (updates: Partial<TrainingConfig>) => void;
  setStatus: (status: Partial<TrainingSimState>) => void;
  pushMetrics: (epoch: number, metrics: TrainingMetrics) => void;
  pushWarning: (warning: any) => void;
}

export const useTrainingSimStore = create<TrainingSimState>((set, get) => ({
  config: {
    epochs: 50,
    batch_size: 16,
    learning_rate: 0.01,
    optimizer: "adam",
    loss_function: "bce",
    l2_lambda: 0.0,
    dropout_rate: 0.0,
    lr_scheduler: null,
    lr_decay_rate: null,
    lr_step_size: null,
    shuffle: true,
    snapshot_interval: 5,
  },
  isTraining: false,
  isPaused: false,
  currentEpoch: 0,
  totalEpochs: 0,
  currentBatch: 0,
  totalBatches: 0,
  metricsHistory: [],
  warnings: [],
  updateConfig: (updates) => set({ config: { ...get().config, ...updates } }),
  setStatus: (status) => set(status),
  pushMetrics: (epoch, metrics) =>
    set((s) => ({
      metricsHistory: [...s.metricsHistory, metrics],
      currentEpoch: epoch,
    })),
  pushWarning: (warning) => set((s) => ({ warnings: [...s.warnings, warning] })),
}));

