import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { IntegratedGradientsResponse } from "../types/simulator";

interface InterpretState {
  loading: boolean;
  error: string | null;
  targetClass: number;
  nSteps: number;
  method: "integrated_gradients" | "shap" | "lime" | "lrp" | "compare";
  result: IntegratedGradientsResponse | null;
  auxImage: string | null;
  infoMessage: string | null;
  setTargetClass: (value: number) => void;
  setNSteps: (value: number) => void;
  setMethod: (value: "integrated_gradients" | "shap" | "lime" | "lrp" | "compare") => void;
  compute: (graphId: string, input: number[], inputShape?: number[] | null) => Promise<void>;
  clear: () => void;
}

export const useInterpretStore = create<InterpretState>((set, get) => ({
  loading: false,
  error: null,
  targetClass: 0,
  nSteps: 50,
  method: "integrated_gradients",
  result: null,
  auxImage: null,
  infoMessage: null,
  setTargetClass: (value) => set({ targetClass: value }),
  setNSteps: (value) => set({ nSteps: value }),
  setMethod: (value) => set({ method: value, infoMessage: null, result: null, auxImage: null }),
  async compute(graphId, input, inputShape) {
    const { targetClass, nSteps, method } = get();
    set({ loading: true, error: null, infoMessage: null, result: null, auxImage: null });
    try {
      if (method === "integrated_gradients") {
        const res = await simulatorApi.integratedGradients(graphId, input, targetClass, nSteps);
        set({ result: res, loading: false });
        return;
      }
      if (method === "shap") {
        const res = await simulatorApi.interpretShap(graphId, input, targetClass, 50);
        set({ loading: false, auxImage: res.shap_base64, infoMessage: "SHAP computed." });
        return;
      }
      if (method === "lime") {
        const res = await simulatorApi.interpretLime(graphId, input, inputShape ?? null, targetClass, 200);
        set({ loading: false, auxImage: res.explanation_image_base64, infoMessage: "LIME computed." });
        return;
      }
      if (method === "lrp") {
        const res = await simulatorApi.interpretLrp(graphId, input, targetClass, "epsilon", 1e-2);
        set({ loading: false, auxImage: res.input_relevance_base64, infoMessage: "LRP computed." });
        return;
      }
      const res = await simulatorApi.interpretCompare(graphId, input, targetClass, ["gradient", "integrated_gradients", "shap", "lime", "lrp"]);
      set({ loading: false, infoMessage: res.message });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Failed to compute interpretability." });
    }
  },
  clear() {
    set({ result: null, error: null, infoMessage: null, auxImage: null });
  },
}));
