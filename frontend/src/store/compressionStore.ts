import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { PruneResponse, QuantizeResponse, PruneSweepResponse } from "../types/simulator";

interface CompressionState {
  loading: boolean;
  error: string | null;
  sparsity: number;
  targetDtype: "int8" | "fp16";
  prune: PruneResponse | null;
  quant: QuantizeResponse | null;
  sweep: PruneSweepResponse | null;
  setSparsity: (value: number) => void;
  setTargetDtype: (value: "int8" | "fp16") => void;
  runPrune: (graphId: string) => Promise<void>;
  runQuant: (graphId: string) => Promise<void>;
  runSweep: (graphId: string, range: number[]) => Promise<void>;
}

export const useCompressionStore = create<CompressionState>((set, get) => ({
  loading: false,
  error: null,
  sparsity: 0.5,
  targetDtype: "int8",
  prune: null,
  quant: null,
  sweep: null,
  setSparsity: (value) => set({ sparsity: value }),
  setTargetDtype: (value) => set({ targetDtype: value }),
  async runPrune(graphId) {
    const { sparsity } = get();
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.compressPrune(graphId, sparsity);
      set({ prune: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Pruning failed." });
    }
  },
  async runQuant(graphId) {
    const { targetDtype } = get();
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.compressQuantize(graphId, targetDtype);
      set({ quant: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Quantization failed." });
    }
  },
  async runSweep(graphId, range) {
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.compressSweep(graphId, range);
      set({ sweep: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Sweep failed." });
    }
  },
}));
