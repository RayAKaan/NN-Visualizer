import { useCallback, useRef, useState } from "react";
import { AnyPredictionResult, ModelType, NeuralState } from "../types";

export function usePrediction(apiBase: string) {
  const [result, setResult] = useState<AnyPredictionResult | null>(null);
  const [state3D, setState3D] = useState<NeuralState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inferenceTime, setInferenceTime] = useState(0);
  const debounceRef = useRef<number>(0);

  const predict = useCallback(
    (pixels: number[], modelType: ModelType = "ann") => {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = window.setTimeout(async () => {
        const start = performance.now();
        setLoading(true);
        setError(null);
        try {
          const [predictRes, stateRes] = await Promise.all([
            fetch(`${apiBase}/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ pixels, model_type: modelType }),
            }),
            fetch(`${apiBase}/state`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ pixels, model_type: modelType }),
            }),
          ]);

          if (!predictRes.ok) throw new Error(`Predict failed: HTTP ${predictRes.status}`);
          if (!stateRes.ok) throw new Error(`State failed: HTTP ${stateRes.status}`);

          setResult((await predictRes.json()) as AnyPredictionResult);
          setState3D((await stateRes.json()) as NeuralState);
          setInferenceTime(performance.now() - start);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Prediction failed");
        } finally {
          setLoading(false);
        }
      }, 50);
    },
    [apiBase]
  );

  const clear = useCallback(() => {
    setResult(null);
    setState3D(null);
    setError(null);
    setInferenceTime(0);
  }, []);

  return { result, state3D, loading, error, inferenceTime, predict, clear };
}
