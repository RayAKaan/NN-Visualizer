import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { KERNEL_PRESETS } from "../data/kernelPresets";
import { useLabStore } from "./labStore";
import { apiClient } from "../api/client";
import { ACTIVATION_RAMP_DARK, ACTIVATION_RAMP_LIGHT, renderHeatmap } from "../utils/colorRamps";

interface KernelLabStore {
  isOpen: boolean;
  activeKernel: number[];
  kernelSize: number;
  targetLayerId: string;
  previewFeatureMap: string | null;
  isPreviewLoading: boolean;
  selectedPresetId: string | null;
  history: number[][];
  historyIndex: number;
  learnedKernel: number[] | null;
  showComparison: boolean;
  _previewTimer: ReturnType<typeof setTimeout> | null;

  openKernelLab: (layerId: string, currentKernel?: number[]) => void;
  closeKernelLab: () => void;
  setCellValue: (row: number, col: number, value: number) => void;
  setKernelSize: (size: 3 | 5 | 7) => void;
  loadPreset: (presetId: string) => void;
  clearKernel: () => void;
  normalizeKernel: () => void;
  invertKernel: () => void;
  rotateKernel: (degrees: 90 | 180 | 270) => void;
  undo: () => void;
  redo: () => void;
  applyToNetwork: () => Promise<void>;
  fetchPreview: () => Promise<void>;
  toggleComparison: () => void;
  fetchLearnedKernel: (layerId: string, kernelIndex: number) => Promise<void>;
}

export const useKernelLabStore = create<KernelLabStore>()(
  devtools((set, get) => ({
    isOpen: false,
    activeKernel: new Array(9).fill(0),
    kernelSize: 3,
    targetLayerId: "",
    previewFeatureMap: null,
    isPreviewLoading: false,
    selectedPresetId: null,
    history: [new Array(9).fill(0)],
    historyIndex: 0,
    learnedKernel: null,
    showComparison: false,
    _previewTimer: null,

    openKernelLab(layerId, currentKernel) {
      const size = currentKernel ? (Math.sqrt(currentKernel.length) as 3 | 5 | 7) : 3;
      const kernel = currentKernel ?? new Array(size * size).fill(0);
      set({
        isOpen: true,
        targetLayerId: layerId,
        activeKernel: [...kernel],
        kernelSize: size,
        history: [[...kernel]],
        historyIndex: 0,
        selectedPresetId: null,
        previewFeatureMap: null,
        learnedKernel: currentKernel ? [...currentKernel] : null,
      });
      void get().fetchPreview();
    },

    closeKernelLab() {
      const t = get()._previewTimer;
      if (t) clearTimeout(t);
      set({ isOpen: false, _previewTimer: null });
    },

    setCellValue(row, col, value) {
      const { activeKernel, kernelSize, history, historyIndex, _previewTimer } = get();
      const next = [...activeKernel];
      next[row * kernelSize + col] = value;
      const nextHistory = history.slice(0, historyIndex + 1);
      nextHistory.push([...next]);
      set({
        activeKernel: next,
        history: nextHistory,
        historyIndex: nextHistory.length - 1,
        selectedPresetId: null,
      });
      if (_previewTimer) clearTimeout(_previewTimer);
      const timer = setTimeout(() => void get().fetchPreview(), 300);
      set({ _previewTimer: timer });
    },

    setKernelSize(size) {
      const kernel = new Array(size * size).fill(0);
      set({ kernelSize: size, activeKernel: kernel, history: [[...kernel]], historyIndex: 0, selectedPresetId: null });
      void get().fetchPreview();
    },

    loadPreset(presetId) {
      const preset = KERNEL_PRESETS.find((p) => p.id === presetId);
      if (!preset) return;
      const { history, historyIndex } = get();
      const nextHistory = history.slice(0, historyIndex + 1);
      nextHistory.push([...preset.values]);
      set({
        activeKernel: [...preset.values],
        kernelSize: preset.size,
        selectedPresetId: presetId,
        history: nextHistory,
        historyIndex: nextHistory.length - 1,
      });
      void get().fetchPreview();
    },

    clearKernel() {
      const k = new Array(get().kernelSize * get().kernelSize).fill(0);
      set({ activeKernel: k, selectedPresetId: null });
      void get().fetchPreview();
    },

    normalizeKernel() {
      const kernel = get().activeKernel;
      const sum = kernel.reduce((acc, v) => acc + Math.abs(v), 0) || 1;
      set({ activeKernel: kernel.map((v) => v / sum), selectedPresetId: null });
      void get().fetchPreview();
    },

    invertKernel() {
      set((s) => ({ activeKernel: s.activeKernel.map((v) => -v), selectedPresetId: null }));
      void get().fetchPreview();
    },

    rotateKernel(degrees) {
      const { activeKernel, kernelSize } = get();
      const iterations = degrees / 90;
      let current = [...activeKernel];
      for (let r = 0; r < iterations; r += 1) {
        const rotated = new Array(kernelSize * kernelSize).fill(0);
        for (let i = 0; i < kernelSize; i += 1) {
          for (let j = 0; j < kernelSize; j += 1) {
            rotated[j * kernelSize + (kernelSize - 1 - i)] = current[i * kernelSize + j];
          }
        }
        current = rotated;
      }
      set({ activeKernel: current, selectedPresetId: null });
      void get().fetchPreview();
    },

    undo() {
      const { history, historyIndex } = get();
      if (historyIndex <= 0) return;
      const next = historyIndex - 1;
      set({ historyIndex: next, activeKernel: [...history[next]] });
      void get().fetchPreview();
    },

    redo() {
      const { history, historyIndex } = get();
      if (historyIndex >= history.length - 1) return;
      const next = historyIndex + 1;
      set({ historyIndex: next, activeKernel: [...history[next]] });
      void get().fetchPreview();
    },

    async applyToNetwork() {
      const { architecture, dataset, inputPixels, triggerReInference } = useLabStore.getState();
      await apiClient.post("/api/lab/apply-kernel", {
        architecture,
        dataset,
        layerId: get().targetLayerId,
        kernel: get().activeKernel,
        pixels: Array.from(inputPixels),
      });
      await triggerReInference();
    },

    async fetchPreview() {
      const { architecture, dataset, inputPixels } = useLabStore.getState();
      const { activeKernel, targetLayerId } = get();
      set({ isPreviewLoading: true });
      try {
        const response = await apiClient.post("/api/lab/kernel-preview", {
          architecture,
          dataset,
          layerId: targetLayerId,
          kernel: activeKernel,
          pixels: Array.from(inputPixels),
        });
        const data = response.data;
        const featureMap = new Float32Array(data.feature_map ?? []);
        const isLight = document.documentElement.classList.contains("light");
        const ramp = isLight ? ACTIVATION_RAMP_LIGHT : ACTIVATION_RAMP_DARK;
        const url = renderHeatmap(featureMap, data.width ?? 28, data.height ?? 28, ramp, true);
        set({ previewFeatureMap: url, isPreviewLoading: false });
      } catch {
        set({ isPreviewLoading: false });
      }
    },

    toggleComparison() {
      set((s) => ({ showComparison: !s.showComparison }));
    },

    async fetchLearnedKernel(layerId, kernelIndex) {
      const { architecture, dataset } = useLabStore.getState();
      const response = await apiClient.get(
        `/api/lab/kernel?arch=${architecture}&dataset=${dataset}&layer=${layerId}&index=${kernelIndex}`,
      );
      set({ learnedKernel: Array.isArray(response.data?.kernel) ? response.data.kernel : null });
    },
  })),
);
