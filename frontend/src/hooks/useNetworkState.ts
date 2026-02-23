import { useRef, useState } from "react";
import { ModelType, NeuralState } from "../types";

export function useNetworkState() {
  const [state, setState] = useState<NeuralState | null>(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef<number | null>(null);
  const fetchState = (pixels: number[], model_type: ModelType) => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setLoading(true);
      const data = await fetch("http://localhost:8000/state", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pixels, model_type }) }).then(r => r.json());
      setState(data); setLoading(false);
    }, 150);
  };
  return { state, loading, fetchState, clear: () => setState(null) };
}
