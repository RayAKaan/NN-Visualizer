import { create } from "zustand";
import type { BindingState, MathVariable } from "../types/binding";

interface BindingStore extends BindingState {
  enabled: boolean;
  variables: Record<string, MathVariable>;
  toggleEnabled: () => void;
  activateFromMath: (variableId: string) => void;
  activateFromVisual: (variableId: string) => void;
  deactivate: () => void;
  registerVariable: (variable: MathVariable) => void;
  unregisterVariable: (variableId: string) => void;
}

export const useBindingStore = create<BindingStore>((set) => ({
  enabled: false,
  activeVariable: null,
  hoveredSource: null,
  connections: [],
  variables: {},
  toggleEnabled: () => set((s) => ({ enabled: !s.enabled, activeVariable: null, hoveredSource: null, connections: [] })),
  activateFromMath: (variableId) => set({ activeVariable: variableId, hoveredSource: "math" }),
  activateFromVisual: (variableId) => set({ activeVariable: variableId, hoveredSource: "visual" }),
  deactivate: () => set({ activeVariable: null, hoveredSource: null, connections: [] }),
  registerVariable: (variable) => set((s) => ({ variables: { ...s.variables, [variable.id]: variable } })),
  unregisterVariable: (variableId) =>
    set((s) => {
      const next = { ...s.variables };
      delete next[variableId];
      return {
        variables: next,
        activeVariable: s.activeVariable === variableId ? null : s.activeVariable,
      };
    }),
}));
