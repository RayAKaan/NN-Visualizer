import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { simulatorApi } from "../hooks/useSimulatorApi";
import { useSimulatorStore } from "./simulatorStore";

interface DebugState {
  overallHealth: "healthy" | "warning" | "critical";
  issues: any[];
  gradientFlow: any | null;
  neuronHealth: any | null;
  lossHealth: any | null;
  warningHistory: any[];
  runDiagnostics: () => Promise<void>;
  applyFix: (fix: any) => Promise<void>;
  clearWarnings: () => void;
}

export const useDebugStore = create<DebugState>()(
  devtools((set, get) => ({
    overallHealth: "healthy",
    issues: [],
    gradientFlow: null,
    neuronHealth: null,
    lossHealth: null,
    warningHistory: [],

    async runDiagnostics() {
      const graphId = useSimulatorStore.getState().graphId;
      if (!graphId) return;
      const res = await simulatorApi.debugDiagnose(graphId);
      set({
        overallHealth: res.overall_health ?? "healthy",
        issues: res.issues ?? [],
        gradientFlow: res.gradient_flow ?? null,
        neuronHealth: res.neuron_health ?? null,
        lossHealth: res.loss_health ?? null,
      });
      if (res.issues?.length) set((s) => ({ warningHistory: [...s.warningHistory, ...res.issues] }));
    },
    async applyFix(fix) {
      const graphId = useSimulatorStore.getState().graphId;
      if (!graphId) return;
      await simulatorApi.debugApplyFix(graphId, fix);
      await get().runDiagnostics();
    },
    clearWarnings() {
      set({ warningHistory: [] });
    },
  })),
);

