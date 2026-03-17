import { useMemo } from "react";
import { renderHeatmap, ACTIVATION_RAMP_DARK } from "../utils/colorRamps";
import type { StageActivation, StageDefinition } from "../types/pipeline";
import type { FlowSnapshot } from "../types/flow";

export function useDataFlow(stages: StageDefinition[], activations: Record<string, StageActivation>): FlowSnapshot[] {
  return useMemo(() => {
    const inputSize = stages[0]?.inputShape?.reduce((a, b) => a * b, 1) || 1;

    return stages
      .filter((s) => Boolean(activations[s.id]))
      .map((stage, stageIndex) => {
        const act = activations[stage.id];
        const data = act.outputData;
        const shape = act.metadata.outputShape.length ? act.metadata.outputShape : stage.outputShape;

        const dimensionality: FlowSnapshot["dimensionality"] =
          shape.length >= 3 ? "3d" : shape.length === 2 ? "2d" : "1d";

        let width = 1;
        let height = 1;
        let payload = data;

        if (dimensionality === "3d") {
          const [, h, w] = shape;
          width = Math.max(1, w || 1);
          height = Math.max(1, h || 1);
          payload = data.slice(0, width * height);
        } else if (dimensionality === "2d") {
          const [h, w] = shape;
          width = Math.max(1, w || 1);
          height = Math.max(1, h || 1);
          payload = data.slice(0, width * height);
        } else {
          width = Math.max(1, Math.min(128, data.length));
          height = 1;
          payload = data.slice(0, width);
        }

        const values = Array.from(payload);
        let mean = 0;
        let sqSum = 0;
        let min = Number.POSITIVE_INFINITY;
        let max = Number.NEGATIVE_INFINITY;
        let zeros = 0;

        for (let i = 0; i < values.length; i += 1) {
          const v = values[i];
          mean += v;
          sqSum += v * v;
          if (v < min) min = v;
          if (v > max) max = v;
          if (v === 0) zeros += 1;
        }
        const n = Math.max(values.length, 1);
        mean /= n;
        const std = Math.sqrt(Math.max(0, sqSum / n - mean * mean));
        const sparsity = zeros / n;

        const bins = new Array<number>(20).fill(0);
        const range = max - min || 1;
        for (let i = 0; i < values.length; i += 1) {
          const bi = Math.min(19, Math.floor(((values[i] - min) / range) * 20));
          bins[bi] += 1;
        }
        let entropy = 0;
        for (let i = 0; i < bins.length; i += 1) {
          const p = bins[i] / n;
          if (p > 0) entropy -= p * Math.log2(p);
        }

        const currentSize = Math.max(1, shape.reduce((a, b) => a * b, 1));
        const dimensionalReduction = currentSize / inputSize;

        let dominantPattern = "mixed";
        if (sparsity > 0.5) dominantPattern = "sparse";
        else if (std < Math.max(1e-6, Math.abs(mean)) * 0.1) dominantPattern = "uniform";
        else if (max > mean + 3 * std) dominantPattern = "peaked";

        return {
          stageId: stage.id,
          stageIndex,
          shape,
          dimensionality,
          thumbnail: {
            url: renderHeatmap(new Float32Array(payload), width, height, ACTIVATION_RAMP_DARK),
            width,
            height,
          },
          statistics: {
            mean,
            std,
            min: Number.isFinite(min) ? min : 0,
            max: Number.isFinite(max) ? max : 0,
            sparsity,
            entropy,
            dimensionalReduction,
          },
          dominantPattern,
        };
      });
  }, [activations, stages]);
}
