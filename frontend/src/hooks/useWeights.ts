import { useState, useEffect } from "react";
import { ModelType } from "../types";

const API = "http://localhost:8000";

export function useWeights(modelType: ModelType) {
  const [weights, setWeights] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    fetch(`${API}/weights?model_type=${modelType}`)
      .then((r) => r.json())
      .then(setWeights)
      .catch(() => setWeights(null));
  }, [modelType]);

  return weights;
}
