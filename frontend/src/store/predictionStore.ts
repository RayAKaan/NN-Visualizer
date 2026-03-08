import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { apiClient } from "../api/client";

export type Architecture = "ANN" | "CNN" | "RNN";
export type ViewMode = "bars" | "radial" | "terrain";

export interface PredictionResult {
  probs: number[];
  label: number;
  confidence: number;
  latencyMs: number;
  activations?: Record<string, any>;
}

export interface ComparisonState {
  inputPixels: Uint8Array;
  results: Record<Architecture, PredictionResult | null>;
  loading: Record<Architecture, boolean>;
  viewMode: ViewMode;
  showTrace: boolean;
  showCompareMode: boolean;
  runId: number;
  runPrediction: () => Promise<void>;
  setInputPixels: (pixels: Uint8Array) => void;
  toggleViewMode: (mode: ViewMode) => void;
  toggleTrace: (show: boolean) => void;
  setCompareMode: (enabled: boolean) => void;
  clearResults: () => void;
}

const ARCHS: Architecture[] = ["ANN", "CNN", "RNN"];
const API_MODEL: Record<Architecture, "ann" | "cnn" | "rnn"> = {
  ANN: "ann",
  CNN: "cnn",
  RNN: "rnn",
};

const inFlight = new Map<Architecture, AbortController>();

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function extractActivations(arch: Architecture, data: any) {
  if (arch === "ANN") {
    const layers = data?.layers && typeof data.layers === "object" ? data.layers : {};
    return {
      layers: [
        { id: "hidden1", name: "H1", type: "dense", values: Array.isArray(layers.hidden1) ? layers.hidden1 : [] },
        { id: "hidden2", name: "H2", type: "dense", values: Array.isArray(layers.hidden2) ? layers.hidden2 : [] },
        { id: "hidden3", name: "H3", type: "dense", values: Array.isArray(layers.hidden3) ? layers.hidden3 : [] },
      ],
    };
  }
  if (arch === "CNN") {
    const fm = Array.isArray(data?.feature_maps) ? data.feature_maps : [];
    const kernels = Array.isArray(data?.kernels) ? data.kernels : [];
    return {
      layers: fm.map((layer: any) => ({
        id: String(layer.layer_name ?? "conv"),
        name: String(layer.layer_name ?? "conv"),
        type: "conv",
        maps: Array.isArray(layer.feature_maps)
          ? layer.feature_maps.map((m: number[], idx: number) => ({
              mapId: idx,
              data: m,
              mean: Array.isArray(layer.mean_activations) ? Number(layer.mean_activations[idx] ?? 0) : 0,
              kernel: Array.isArray(kernels) && kernels.length > 0 ? kernels[0]?.kernels?.[0]?.flat?.() ?? [] : [],
            }))
          : [],
      })),
    };
  }
  const steps = Array.isArray(data?.timestep_activations) ? data.timestep_activations : [];
  const lstm = Array.isArray(data?.lstm_output) ? data.lstm_output : [];
  const makeGate = (seed: number) => clamp01(Math.abs(Math.sin(seed)));
  return {
    layers: [{ id: "lstm", name: "LSTM", type: "lstm" }],
    timesteps: steps.map((v: number, i: number) => {
      const local = lstm.slice(i * 2, i * 2 + 8);
      const hidden = local.length > 0 ? local : [v];
      return {
        hidden,
        gates: {
          f: [makeGate(v + i * 0.11)],
          i: [makeGate(v + i * 0.17 + 0.8)],
          o: [makeGate(v + i * 0.07 + 1.3)],
        },
        attention: clamp01(Math.abs(v)),
      };
    }),
  };
}

export const useComparisonStore = create<ComparisonState>()(
  devtools((set, get) => ({
    inputPixels: new Uint8Array(28 * 28),
    results: { ANN: null, CNN: null, RNN: null },
    loading: { ANN: false, CNN: false, RNN: false },
    viewMode: "bars",
    showTrace: false,
    showCompareMode: true,
    runId: 0,

    async runPrediction() {
      const { inputPixels } = get();
      const active = Array.from(inputPixels).filter((v) => v > 25).length / inputPixels.length;
      if (active < 0.05) {
        set({ results: { ANN: null, CNN: null, RNN: null } });
        return;
      }
      const payload = { pixels: Array.from(inputPixels).map((v) => v / 255) };
      ARCHS.forEach((arch) => {
        inFlight.get(arch)?.abort();
        const ctl = new AbortController();
        inFlight.set(arch, ctl);
        set((s) => ({ loading: { ...s.loading, [arch]: true } }));
      });
      const runId = get().runId + 1;
      set({ runId });

      const jobs = ARCHS.map(async (arch) => {
        const started = performance.now();
        const ctl = inFlight.get(arch)!;
        const res = await apiClient.post(
          "/predict",
          { ...payload, model_type: API_MODEL[arch] },
          { signal: ctl.signal },
        );
        const data = res.data ?? {};
        const probs = Array.isArray(data.probabilities) ? data.probabilities.map((v: unknown) => (typeof v === "number" ? v : 0)) : [];
        const label =
          probs.length > 0
            ? probs.reduce((i: number, v: number, j: number) => (v > probs[i] ? j : i), 0)
            : 0;
        return {
          arch,
          result: {
            probs: probs.length > 0 ? probs : Array(10).fill(0),
            label,
            confidence: Number(probs[label] ?? data.confidence ?? 0),
            latencyMs: Math.round(performance.now() - started),
            activations: extractActivations(arch, data),
          } as PredictionResult,
        };
      });

      const settled = await Promise.allSettled(jobs);
      if (get().runId !== runId) return;
      settled.forEach((r) => {
        if (r.status === "fulfilled") {
          const { arch, result } = r.value;
          set((s) => ({
            results: { ...s.results, [arch]: result },
            loading: { ...s.loading, [arch]: false },
          }));
        }
      });
      ARCHS.forEach((arch) => set((s) => ({ loading: { ...s.loading, [arch]: false } })));
    },

    setInputPixels(pixels) {
      set({ inputPixels: pixels });
    },
    toggleViewMode(mode) {
      set({ viewMode: mode });
    },
    toggleTrace(show) {
      set({ showTrace: show });
    },
    setCompareMode(enabled) {
      set({ showCompareMode: enabled });
    },
    clearResults() {
      set({ results: { ANN: null, CNN: null, RNN: null } });
    },
  })),
);
