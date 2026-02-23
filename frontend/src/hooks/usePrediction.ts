import { useRef, useState } from "react";
import { AnyPredictionResult, ModelType } from "../types";

export function usePrediction() {
  const [result, setResult] = useState<AnyPredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  const predict = (pixels: number[], model_type: ModelType) => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      const t0 = performance.now();
      setLoading(true); setError(null);
      try {
        const res = await fetch("http://localhost:8000/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pixels, model_type }) });
        const data = await res.json();
        setResult(data);
        setInferenceTime(performance.now() - t0);
      } catch (e) {
        setError((e as Error).message);
      } finally { setLoading(false); }
    }, 120);
  };

  return { result, loading, error, inferenceTime, predict, clear: () => setResult(null) };
}
