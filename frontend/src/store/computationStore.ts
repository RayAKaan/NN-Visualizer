import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { ActivationInspection, EquationResponse, ForwardStep, WeightInspection } from "../types/simulator";
import { simulatorApi } from "../hooks/useSimulatorApi";
import { useSimulatorStore } from "./simulatorStore";

interface ComputationState {
  steps: ForwardStep[];
  layerOutputs: Record<string, number[]>;
  equations: EquationResponse | null;
  weightInspection: WeightInspection | null;
  activationInspection: ActivationInspection | null;
  runFullForward: (input: number[]) => Promise<void>;
  runStepForward: (stepIndex: number) => Promise<void>;
  fetchEquations: (layerIndex: number) => Promise<void>;
  inspectWeights: (layerIndex: number) => Promise<void>;
  inspectActivations: (layerIndex: number, input: number[]) => Promise<void>;
  reset: () => void;
}

export const useComputationStore = create<ComputationState>()(
  devtools((set, get) => ({
    steps: [],
    layerOutputs: {},
    equations: null,
    weightInspection: null,
    activationInspection: null,

    async runFullForward(input) {
      const { graphId } = useSimulatorStore.getState();
      if (!graphId) return;
      const res = await simulatorApi.forwardFull(graphId, input);
      set({ steps: res.steps, layerOutputs: res.layer_outputs });
      useSimulatorStore.getState().setForwardMeta({
        forwardPassState: "complete",
        currentStepIndex: 0,
        totalSteps: res.total_steps,
      });
    },
    async runStepForward(stepIndex) {
      const { graphId, currentInput } = useSimulatorStore.getState();
      if (!graphId || !currentInput) return;
      const res = await simulatorApi.forwardStep(graphId, currentInput, stepIndex);
      set({ steps: [...get().steps.slice(0, stepIndex + 1), res.step] });
      useSimulatorStore.getState().setForwardMeta({
        forwardPassState: "stepping",
        currentStepIndex: stepIndex,
      });
    },
    async fetchEquations(layerIndex) {
      const { graphId } = useSimulatorStore.getState();
      if (!graphId) return;
      const res = await simulatorApi.equationsLayer(graphId, layerIndex, true);
      set({ equations: res });
    },
    async inspectWeights(layerIndex) {
      const { graphId } = useSimulatorStore.getState();
      if (!graphId) return;
      const res = await simulatorApi.inspectWeights(graphId, layerIndex);
      set({ weightInspection: res });
    },
    async inspectActivations(layerIndex, input) {
      const { graphId } = useSimulatorStore.getState();
      if (!graphId) return;
      const res = await simulatorApi.inspectActivations(graphId, layerIndex, input);
      set({ activationInspection: res });
    },
    reset() {
      set({ steps: [], layerOutputs: {}, equations: null, weightInspection: null, activationInspection: null });
      useSimulatorStore.getState().setForwardMeta({ forwardPassState: "idle", currentStepIndex: 0, totalSteps: 0 });
    },
  })),
);

