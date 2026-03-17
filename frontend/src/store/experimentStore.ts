import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { ExperimentRecord } from "../types/simulator";

interface ExperimentState {
  loading: boolean;
  experiments: ExperimentRecord[];
  error: string | null;
  fetchAll: () => Promise<void>;
  create: (payload: { name: string; config: Record<string, any> }) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

export const useExperimentStore = create<ExperimentState>((set) => ({
  loading: false,
  experiments: [],
  error: null,
  async fetchAll() {
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.experimentsList();
      set({ experiments: res.experiments, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Failed to load experiments." });
    }
  },
  async create(payload) {
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.experimentsCreate({ name: payload.name, config: payload.config });
      set((s) => ({ experiments: [res, ...s.experiments], loading: false }));
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Failed to create experiment." });
    }
  },
  async remove(id) {
    set({ loading: true, error: null });
    try {
      await simulatorApi.experimentsDelete(id);
      set((s) => ({ experiments: s.experiments.filter((e) => e.id !== id), loading: false }));
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Failed to delete experiment." });
    }
  },
}));
