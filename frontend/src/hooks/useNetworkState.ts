import { useState, useCallback, useRef } from "react";
import { ModelType } from "../types";

const API = "http://localhost:8000";

export function useNetworkState() {
  const [state, setState] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const timer = useRef<number>(0);

  const fetchState = useCallback((pixels: number[], modelType?: ModelType) => {
    clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API}/state`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels, model_type: modelType }),
        });
        if (res.ok) {
          setState(await res.json());
        }
      } catch (e) {
        console.error("Failed to fetch state", e);
      } finally {
        setLoading(false);
      }
    }, 100);
  }, []);

  const clear = useCallback(() => setState(null), []);

  return { state, loading, fetchState, clear };
}
