import { useEffect, useState } from "react";
import { ModelInfo } from "../types";

export function useModelInfo(apiBase: string, modelType?: string) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    const query = modelType ? `?type=${modelType}` : "";
    fetch(`${apiBase}/model/info${query}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setModelInfo(data as ModelInfo);
        }
      })
      .catch(() => setModelInfo(null));
  }, [apiBase, modelType]);

  return modelInfo;
}
