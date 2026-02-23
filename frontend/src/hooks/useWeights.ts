import { useEffect, useState } from "react";
import { ModelType } from "../types";

export function useWeights(modelType: ModelType) {
  const [weights, setWeights] = useState<Record<string, unknown>>({});
  useEffect(() => {
    fetch(`http://localhost:8000/weights?model_type=${modelType}`).then(r => r.json()).then(setWeights).catch(() => setWeights({}));
  }, [modelType]);
  return { weights };
}
