import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { LandscapeResult } from "../types/simulator";

interface LandscapeState {
  taskId: string | null;
  status: "idle" | "computing" | "complete" | "error";
  progress: number;
  resolution: number;
  range: number;
  result: LandscapeResult | null;
  error: string | null;
  setResolution: (value: number) => void;
  setRange: (value: number) => void;
  compute: (graphId: string, datasetId: string) => Promise<void>;
  poll: () => Promise<void>;
  reset: () => void;
}

export const useLandscapeStore = create<LandscapeState>((set, get) => ({
  taskId: null,
  status: "idle",
  progress: 0,
  resolution: 30,
  range: 1.0,
  result: null,
  error: null,
  setResolution: (value) => set({ resolution: value }),
  setRange: (value) => set({ range: value }),
  async compute(graphId, datasetId) {
    const { resolution, range } = get();
    set({ status: "computing", error: null, progress: 0, result: null });
    const res = await simulatorApi.landscapeCompute(graphId, datasetId, resolution, range);
    set({ taskId: res.task_id, status: res.status === "complete" ? "complete" : "computing" });
  },
  async poll() {
    const { taskId, status } = get();
    if (!taskId || status === "complete" || status === "error") return;
    const res = await simulatorApi.landscapeStatus(taskId);
    set({
      status: res.status === "complete" ? "complete" : res.status === "error" ? "error" : "computing",
      progress: res.progress ?? 0,
      result: res.result ?? null,
    });
  },
  reset() {
    set({ taskId: null, status: "idle", progress: 0, result: null, error: null });
  },
}));
