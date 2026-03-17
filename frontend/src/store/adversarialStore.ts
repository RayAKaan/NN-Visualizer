import { create } from "zustand";
import { simulatorApi } from "../hooks/useSimulatorApi";
import type { AdversarialAttackResponse, RobustnessCurveResponse } from "../types/simulator";

interface AdversarialState {
  loading: boolean;
  error: string | null;
  method: "fgsm" | "pgd";
  epsilon: number;
  pgdSteps: number;
  pgdStepSize: number;
  result: AdversarialAttackResponse | null;
  robustness: RobustnessCurveResponse | null;
  setMethod: (value: "fgsm" | "pgd") => void;
  setEpsilon: (value: number) => void;
  setPgdSteps: (value: number) => void;
  setPgdStepSize: (value: number) => void;
  attack: (payload: { graph_id: string; input: number[]; true_label: number; input_shape?: number[] | null }) => Promise<void>;
  evaluate: (payload: { graph_id: string; dataset_id: string; epsilons: number[] }) => Promise<void>;
}

export const useAdversarialStore = create<AdversarialState>((set, get) => ({
  loading: false,
  error: null,
  method: "fgsm",
  epsilon: 0.1,
  pgdSteps: 7,
  pgdStepSize: 0.02,
  result: null,
  robustness: null,
  setMethod: (value) => set({ method: value }),
  setEpsilon: (value) => set({ epsilon: value }),
  setPgdSteps: (value) => set({ pgdSteps: value }),
  setPgdStepSize: (value) => set({ pgdStepSize: value }),
  async attack(payload) {
    const { method, epsilon, pgdSteps, pgdStepSize } = get();
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.adversarialAttack({
        graph_id: payload.graph_id,
        input: payload.input,
        true_label: payload.true_label,
        method,
        epsilon,
        input_shape: payload.input_shape ?? null,
        pgd_steps: pgdSteps,
        pgd_step_size: pgdStepSize,
      });
      set({ result: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Adversarial attack failed." });
    }
  },
  async evaluate(payload) {
    set({ loading: true, error: null });
    try {
      const res = await simulatorApi.adversarialEvaluate({
        graph_id: payload.graph_id,
        dataset_id: payload.dataset_id,
        epsilons: payload.epsilons,
      });
      set({ robustness: res, loading: false });
    } catch (err: any) {
      set({ loading: false, error: err?.message ?? "Robustness evaluation failed." });
    }
  },
}));
