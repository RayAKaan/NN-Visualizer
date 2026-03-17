import { create } from "zustand";

export type SimulatorView = "network" | "playground" | "inspector" | "metrics" | "activations" | "sequence" | "compare" | "profile" | "landscape" | "embeddings" | "interpret" | "adversarial" | "compress" | "generate" | "augment" | "experiments";

interface SimulatorState {
  graphId: string | null;
  isBuilt: boolean;
  activeView: SimulatorView;
  selectedLayerIndex: number;
  forwardPassState: "idle" | "stepping" | "complete";
  currentStepIndex: number;
  totalSteps: number;
  autoPlay: boolean;
  animationSpeed: number;
  currentInput: number[] | null;
  currentTarget: number[] | null;
  setGraphId: (id: string | null) => void;
  setActiveView: (view: SimulatorView) => void;
  setSelectedLayerIndex: (index: number) => void;
  setForwardMeta: (state: Partial<Pick<SimulatorState, "forwardPassState" | "currentStepIndex" | "totalSteps">>) => void;
  setAnimationSpeed: (speed: number) => void;
  setAutoPlay: (autoPlay: boolean) => void;
  setCurrentInput: (input: number[] | null) => void;
  setCurrentTarget: (target: number[] | null) => void;
}

export const useSimulatorStore = create<SimulatorState>((set) => ({
  graphId: null,
  isBuilt: false,
  activeView: "network",
  selectedLayerIndex: 0,
  forwardPassState: "idle",
  currentStepIndex: 0,
  totalSteps: 0,
  autoPlay: false,
  animationSpeed: 1,
  currentInput: null,
  currentTarget: null,
  setGraphId: (id) => set({ graphId: id, isBuilt: Boolean(id) }),
  setActiveView: (view) => set({ activeView: view }),
  setSelectedLayerIndex: (index) => set({ selectedLayerIndex: index }),
  setForwardMeta: (state) => set(state),
  setAnimationSpeed: (speed) => set({ animationSpeed: speed }),
  setAutoPlay: (autoPlay) => set({ autoPlay }),
  setCurrentInput: (input) => set({ currentInput: input }),
  setCurrentTarget: (target) => set({ currentTarget: target }),
}));
