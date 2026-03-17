import { create } from "zustand";
import { apiClient } from "../api/client";
import { buildPipelineStages } from "../data/equations";
import type { Architecture, Dataset, StageActivation, StageDefinition } from "../types/pipeline";

interface ComparisonResult {
  stages: StageDefinition[];
  activations: Record<string, StageActivation>;
  prediction: { label: number | string; confidence: number; probs: number[] };
  totalTimeMs?: number;
}

interface ComparisonStore {
  isComparisonActive: boolean;
  loading: Record<Architecture, boolean>;
  results: Record<Architecture, ComparisonResult | null>;
  startComparison: (pixels: Float32Array, dataset: Dataset) => Promise<void>;
  stopComparison: () => void;
}

function parseStageActivation(data: any, stageId: string): StageActivation {
  return {
    stageId,
    inputData: new Float32Array(data.input ?? []),
    outputData: new Float32Array(data.output ?? []),
    weights: data.weights ? new Float32Array(data.weights) : undefined,
    bias: data.bias ? new Float32Array(data.bias) : undefined,
    kernels: Array.isArray(data.kernels) ? data.kernels.map((k: number[]) => new Float32Array(k)) : undefined,
    gateValues: data.gates
      ? {
          forget: new Float32Array(data.gates.forget ?? []),
          input: new Float32Array(data.gates.input ?? []),
          output: new Float32Array(data.gates.output ?? []),
          cellState: new Float32Array(data.gates.cell_state ?? []),
        }
      : undefined,
    metadata: {
      inputShape: Array.isArray(data.input_shape) ? data.input_shape : [],
      outputShape: Array.isArray(data.output_shape) ? data.output_shape : [],
      paramCount: typeof data.param_count === "number" ? data.param_count : 0,
      computeTimeMs: typeof data.compute_time_ms === "number" ? data.compute_time_ms : 0,
    },
  };
}

export const useComparisonStore = create<ComparisonStore>((set) => ({
  isComparisonActive: false,
  loading: { ANN: false, CNN: false, RNN: false },
  results: { ANN: null, CNN: null, RNN: null },

  async startComparison(pixels, dataset) {
    set({
      isComparisonActive: true,
      loading: { ANN: true, CNN: true, RNN: true },
      results: { ANN: null, CNN: null, RNN: null },
    });

    try {
      const { data } = await apiClient.post("/api/lab/comparison/run-all", {
        pixels: Array.from(pixels),
        dataset,
      });

      const mapped = (["ANN", "CNN", "RNN"] as Architecture[]).reduce<Record<Architecture, ComparisonResult | null>>(
        (acc, arch) => {
          const entry = data?.[arch];
          if (!entry) {
            acc[arch] = null;
            return acc;
          }

          const stages = buildPipelineStages(arch, dataset);
          const acts: Record<string, StageActivation> = {};
          Object.entries<any>(entry.activations ?? {}).forEach(([stageId, raw]) => {
            acts[stageId] = parseStageActivation(raw, stageId);
          });

          acc[arch] = {
            stages,
            activations: acts,
            prediction: entry.prediction,
            totalTimeMs: entry.totalTimeMs,
          };
          return acc;
        },
        { ANN: null, CNN: null, RNN: null },
      );

      set({
        results: mapped,
        loading: { ANN: false, CNN: false, RNN: false },
      });
    } catch {
      set({ loading: { ANN: false, CNN: false, RNN: false } });
    }
  },

  stopComparison() {
    set({
      isComparisonActive: false,
      loading: { ANN: false, CNN: false, RNN: false },
      results: { ANN: null, CNN: null, RNN: null },
    });
  },
}));
