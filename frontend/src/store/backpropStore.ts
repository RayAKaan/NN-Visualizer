import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { BackwardFullResponse, BackwardStep } from "../types/simulator";
import { simulatorApi } from "../hooks/useSimulatorApi";
import { useSimulatorStore } from "./simulatorStore";
import { useTrainingSimStore } from "./trainingSimStore";

interface BackpropState {
  mode: "forward" | "backward" | "full_cycle";
  backwardSteps: BackwardStep[];
  currentBackwardStep: number;
  totalBackwardSteps: number;
  lossValue: number | null;
  lossEquation: string | null;
  gradientSummary: BackwardFullResponse["gradient_summary"] | null;
  runBackward: (input: number[], target: number[]) => Promise<void>;
  stepBackward: (stepIndex: number) => Promise<void>;
  resetBackward: () => void;
  switchMode: (mode: "forward" | "backward" | "full_cycle") => void;
}

export const useBackpropStore = create<BackpropState>()(
  devtools((set, get) => ({
    mode: "forward",
    backwardSteps: [],
    currentBackwardStep: 0,
    totalBackwardSteps: 0,
    lossValue: null,
    lossEquation: null,
    gradientSummary: null,

    async runBackward(input, target) {
      const graphId = useSimulatorStore.getState().graphId;
      const lossFn = useTrainingSimStore.getState().config.loss_function;
      const l2 = useTrainingSimStore.getState().config.l2_lambda;
      if (!graphId) return;
      const res = await simulatorApi.backwardFull(graphId, input, target, lossFn, l2);
      set({
        backwardSteps: res.steps,
        totalBackwardSteps: res.total_steps,
        currentBackwardStep: 0,
        lossValue: res.loss_value,
        lossEquation: res.loss_equation,
        gradientSummary: res.gradient_summary,
      });
    },
    async stepBackward(stepIndex) {
      const graphId = useSimulatorStore.getState().graphId;
      const input = useSimulatorStore.getState().currentInput;
      const target = useSimulatorStore.getState().currentTarget;
      const lossFn = useTrainingSimStore.getState().config.loss_function;
      if (!graphId || !input || !target) return;
      const res = await simulatorApi.backwardStep(graphId, input, target, lossFn, stepIndex);
      const nextSteps = [...get().backwardSteps];
      nextSteps[stepIndex] = res.step;
      set({ backwardSteps: nextSteps, currentBackwardStep: stepIndex });
    },
    resetBackward() {
      set({ backwardSteps: [], currentBackwardStep: 0, totalBackwardSteps: 0, lossValue: null, lossEquation: null, gradientSummary: null });
    },
    switchMode(mode) {
      set({ mode });
    },
  })),
);
